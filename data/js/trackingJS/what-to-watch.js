if (window.__prefetching) {
    throw new Error("No error, prefetch done. Stop execution.")
}
_DM.define("DM_Controller", function() {
    var b = {
        call: function(a, h) {
            var i, g;
            if (h) {
                g = h.callback || function() {};
                i = h.postArgs;
                if ($.isArray(h.args)) {
                    h["method:args"] = $.toJSONString(h.args);
                    delete (h.args)
                } else {
                    delete (h.args)
                }
                delete (h.callback);
                delete (h.postArgs)
            } else {
                h = {}
            }
            if (!h.request) {
                h.request = document.location.pathname
            }
            var j = "/controller/" + a + "?" + $.param(h);
            (i ? $.post(j, i) : $.get(j)).done(g)
        }
    };
    return b
}, true);
_DM.define("Keyboard", function() {
    var b = {
        TAB: 9,
        BACKSPACE: 8,
        ENTER: 13,
        SHIFT: 16,
        CONTROL: 17,
        ESCAPE: 27,
        SPACE: 32,
        ARROW_LEFT: 37,
        ARROW_UP: 38,
        ARROW_RIGHT: 39,
        ARROW_DOWN: 40,
        COMMAND: 91,
        SEMICOLON: 59,
        COMA: 188,
        A: 65,
        B: 66,
        C: 67,
        D: 68,
        E: 69,
        F: 70,
        G: 71,
        H: 72,
        I: 73,
        J: 74,
        K: 75,
        L: 76,
        M: 77,
        N: 78,
        O: 79,
        P: 80,
        Q: 81,
        R: 82,
        S: 83,
        T: 84,
        U: 85,
        V: 86,
        W: 87,
        X: 88,
        Y: 89,
        Z: 90,
        is_an_input_modifier_key: function(a) {
            return [b.ENTER, b.SHIFT, b.CONTROL, b.ESCAPE, b.SPACE, b.ARROW_LEFT, b.ARROW_UP, b.ARROW_RIGHT, b.ARROW_DOWN, b.COMMAND, b.TAB].indexOf(a)===-1
        }
    };
    return b
}, true);
_DM.define("DM_Widget_Page", function() {
    var b = {
        initEvents: function() {
            $(document).on("click", ".js-popup-close", DM_Widget_Page.closePopup);
            $("span.linkable").each(function() {
                b.makeLinkable(this)
            });
            PubSub.subscribe("unlogged_action:done", DM_Widget_Page.onUnloggedActionDone)
        },
        makeLinkable: function(a) {
            var m = $(a), n;
            if (typeof a.outerHTML != "undefined") {
                var l = a.outerHTML;
                l = l.replace(/<span/i, "<a").replace(/<\/span>$/i, "</a>");
                if (m.hasClass("ajax_link")) {
                    l = l.replace("<a", '<a onclick="DM_Ajax.link(this); return false;"')
                }
                var q = document.createElement("div");
                q.innerHTML = l;
                n = $(q.firstChild);
                var p = n.data("href");
                if (p) {
                    n.attr("href", p)
                }
            } else {
                n = $("<a>").html(a.innerHTML);
                for (var o = 0; o < a.attributes.length; o++) {
                    var i = a.attributes[o].value, r = a.attributes[o].name;
                    if (r == "data-href") {
                        r = "href"
                    }
                    n.attr(r, i)
                }
                if (m.hasClass("ajax_link")) {
                    n.attr("onclick", "DM_Ajax.link(this); return false;")
                }
            }
            m.replaceWith(n)
        },
        closePopup: function() {
            PubSub.publish("popup:close")
        },
        notify: function() {
            var a = arguments;
            DM_JS.get("DM_Notify", function() {
                DM_Notify.notify.apply(DM_Notify, Array.prototype.slice.call(a))
            })
        },
        openLoginPopupForAction: function(a, e, f) {
            DM_JS.get("DM_Widget_Popup", function() {
                b.popup = new DM_Widget_Popup({
                    close_button: false
                });
                var c = DM_Widget_Popup.getInstanceOrNew();
                c.loadPageItem("/pageitem/loginfull?restricted=1", document.location.pathname, {
                    unlogged_action: e || ""
                }, function() {
                    $("#popupContent input[data-focus]").focus();
                    store.set("unlogged_action", {
                        name: a,
                        params: f
                    })
                })
            })
        },
        onUnloggedActionDone: function() {
            store.remove("unlogged_action")
        },
        continuePendingActions: function() {
            var d;
            if (typeof store != "undefined" && store.get("unlogged_action")) {
                b.onUnloggedActionDone()
            }
            if (d = getCookie("dm_done")) {
                deleteCookie("dm_done");
                window["DM_Widget_Page_" + d] && DM_Widget_Page.notify(window["DM_Widget_Page_" + d], "info")
            } else {
                if (d = getCookie("dm_need")) {
                    deleteCookie("dm_need");
                    var a = d.split(":");
                    if (a.length > 1 && a[0] == "popup") {
                        DM_JS.get("DM_Widget_Popup", function() {
                            var c = DM_Widget_Popup.getInstanceOrNew();
                            c.loadPageItem(hex2str(a[1]), document.location.pathname)
                        })
                    }
                }
            }
        },
        __initialize: function() {
            PubSub.subscribe("page:notify", DM_Widget_Page.notify);
            PubSub.subscribe("page:login", DM_Widget_Page.openLoginPopupForAction);
            $(function() {
                setCookie("tzo", new Date().getTimezoneOffset() * 60, 1);
                if (window.DM_WrongUserEmail) {
                    $("#topwrapper").prepend(window.DM_WrongUserEmail)
                }
                b.continuePendingActions();
                b.initEvents()
            })
        }
    };
    return b
}, true);
(function(u) {
    var p = {}, q = u.document, y = "localStorage", A = "script", v;
    p.disabled = false;
    p.set = function(b, a) {};
    p.get = function(a) {};
    p.remove = function(a) {};
    p.clear = function() {};
    p.transact = function(d, a, c) {
        var b = p.get(d);
        if (c == null) {
            c = a;
            a = null
        }
        if (typeof b == "undefined") {
            b = a || {}
        }
        c(b);
        p.set(d, b)
    };
    p.getAll = function() {};
    p.forEach = function() {};
    p.serialize = function(a) {
        return JSON.stringify(a)
    };
    p.deserialize = function(b) {
        if (typeof b != "string") {
            return undefined
        }
        try {
            return JSON.parse(b)
        } catch (a) {
            return b || undefined
        }
    };
    function z() {
        try {
            return (y in u && u[y])
        } catch (a) {
            return false
        }
    }
    if (z()) {
        v = u[y];
        p.set = function(b, a) {
            if (a === undefined) {
                return p.remove(b)
            }
            v.setItem(b, p.serialize(a));
            return a
        };
        p.get = function(a) {
            return p.deserialize(v.getItem(a))
        };
        p.remove = function(a) {
            v.removeItem(a)
        };
        p.clear = function() {
            v.clear()
        };
        p.getAll = function() {
            var a = {};
            p.forEach(function(c, b) {
                a[c] = b
            });
            return a
        };
        p.forEach = function(a) {
            for (var c = 0; c < v.length; c++) {
                var b = v.key(c);
                a(b, p.get(b))
            }
        }
    } else {
        if (q.documentElement.addBehavior) {
            var r, w;
            try {
                w = new ActiveXObject("htmlfile");
                w.open();
                w.write("<" + A + ">document.w=window</" + A + '><iframe src="/favicon.ico"></iframe>');
                w.close();
                r = w.w.frames[0].document;
                v = r.createElement("div")
            } catch (s) {
                v = q.createElement("div");
                r = q.body
            }
            function B(a) {
                return function() {
                    var b = Array.prototype.slice.call(arguments, 0);
                    b.unshift(v);
                    r.appendChild(v);
                    v.addBehavior("#default#userData");
                    v.load(y);
                    var c = a.apply(p, b);
                    r.removeChild(v);
                    return c
                }
            }
            var x = new RegExp("[!\"#$%&'()*+,/\\\\:;<=>?@[\\]^`{|}~]", "g");
            function e(a) {
                return a.replace(/^d/, "___$&").replace(x, "___")
            }
            p.set = B(function(a, c, b) {
                c = e(c);
                if (b === undefined) {
                    return p.remove(c)
                }
                a.setAttribute(c, p.serialize(b));
                a.save(y);
                return b
            });
            p.get = B(function(b, a) {
                a = e(a);
                return p.deserialize(b.getAttribute(a))
            });
            p.remove = B(function(b, a) {
                a = e(a);
                b.removeAttribute(a);
                b.save(y)
            });
            p.clear = B(function(a) {
                var d = a.XMLDocument.documentElement.attributes;
                a.load(y);
                for (var b = 0, c; c = d[b]; b++) {
                    a.removeAttribute(c.name)
                }
                a.save(y)
            });
            p.getAll = function(b) {
                var a = {};
                p.forEach(function(d, c) {
                    a[d] = c
                });
                return a
            };
            p.forEach = B(function(a, b) {
                var f = a.XMLDocument.documentElement.attributes;
                for (var c = 0, d; d = f[c]; ++c) {
                    b(d.name, p.deserialize(a.getAttribute(d.name)))
                }
            })
        }
    }
    try {
        var t = "__storejs__";
        p.set(t, t);
        if (p.get(t) != t) {
            p.disabled = true
        }
        p.remove(t)
    } catch (s) {
        p.disabled = true
    }
    p.enabled=!p.disabled;
    if (typeof module != "undefined" && module.exports && this.module !== module) {
        module.exports = p
    } else {
        if (typeof define === "function" && define.amd) {
            define(p)
        } else {
            u.store = p
        }
    }
})(Function("return this")());
function iOSversion() {
    if (/iP(hone|od|ad)/.test(navigator.platform)) {
        var b = (navigator.appVersion).match(/OS (\d+)_(\d+)_?(\d+)?/);
        return [parseInt(b[1], 10), parseInt(b[2], 10), parseInt(b[3] || 0, 10)]
    }
    return false
}
function safariVersion() {
    if (/Safari/.test(navigator.userAgent)) {
        return [0, 0, 0]
    }
    return false
}
var ver = iOSversion() || safariVersion();
if (!store.enabled && ver && ver[0] <= 8) {
    window.fakeStorage = {};
    store.get = function(b) {
        return this.deserialize(fakeStorage[b])
    };
    store.set = function(d, c) {
        fakeStorage[d] = this.serialize(c)
    };
    store.clear = function() {
        fakeStorage = {}
    };
    store.serialize = function(b) {
        return JSON.stringify(b)
    };
    store.deserialize = function(d) {
        if (typeof d != "string") {
            return undefined
        }
        try {
            return JSON.parse(d)
        } catch (c) {
            return d || undefined
        }
    }
}
_DM.define("storeWithExpiration", function() {
    var b = {
        set: function(a, e, f) {
            store.set(a, {
                val: e,
                exp: f,
                time: new Date().getTime()
            })
        },
        get: function(a) {
            var d = store.get(a);
            if (!d) {
                return null
            }
            if (new Date().getTime() - d.time > d.exp) {
                return null
            }
            return d.val
        },
        getNewExp: function(a) {
            var d = store.get(a);
            if (!d) {
                return null
            }
            if (new Date().getTime() - d.time > d.exp) {
                return null
            }
            return d.exp - (new Date().getTime() - d.time)
        }
    };
    return b
}, true);
(function(h) {
    var f, g, e = [];
    h.extend({
        loadPickadate: function(c, a) {
            if (f) {
                a();
                return this
            }
            e.push(a);
            if (g) {
                return this
            }
            DM_WidgetV3.addCss("@pickadatetime." + DM_Context.direction + ".css");
            g = true;
            var m = ["bg_BG", "ca_ES", "da_DK", "el_GR", "et_EE", "fi_FI", "he_IL", "hu_HU", "it_IT", "no_NO", "pt_BR", "ro_RO", "sk_SK", "th_TH", "uk_UA", "bs_BA", "cs_CZ", "de_DE", "es_ES", "eu_ES", "fr_FR", "hr_HR", "id_ID", "nl_NL", "pl_PL", "pt_PT", "ru_RU", "sv_SE", "tr_TR", "zh_CN", "vi_VN", "ja_JP", "ko_KR"], p, d;
            if (m.indexOf(c) >= 0) {
                p = c
            } else {
                var b = c.slice(0, 2);
                if (m.indexOf(b + "_" + b.toUpperCase())) {
                    p = b + "_" + b.toUpperCase()
                }
            }
            if (!p || p.match(/^en_/) || m.indexOf(p) < 0) {
                p = "default"
            } else {
                d = "pickadate_" + p
            }
            function o() {
                g = false;
                f = true;
                for (var j = 0, i = e.length; j < i; j++) {
                    e[j]()
                }
            }
            function n() {
                if (p != "default"&&!window[d]) {
                    DM_JS.load("/js/lib/plugins/jquery.pickadate/translations/" + p + ".js", d, function() {
                        window[d] = true;
                        o()
                    })
                } else {
                    o()
                }
            }
            if (!h.fn.pickadate ||!h.fn.pickatime) {
                DM_JS.load("@pickadatetime.js", "Pickadate", function() {
                    n()
                })
            } else {
                n()
            }
        }
    })
})(jQuery);
_DM.define("DM_Form_Utils", function() {
    var b = {
        persistData: function(a) {
            store.set("form:" + a.find("#form_name").val(), a.serializeJSON())
        },
        applyPersistentData: function(a) {
            var f = "form:" + a.find("#form_name").val(), g = store.get(f), h = false;
            if (g) {
                $.each(g, function(d, c) {
                    $("#" + d).val(c);
                    h = true
                });
                store.remove(f)
            }
            return h
        }
    };
    return b
}, true);
_DM.define("DM_Form_Input", function() {
    var b = function(l, h) {
        this.observers = [];
        this.dm_form = l;
        this._metaInputElement = h;
        this.observedInputs = {};
        this.changeObserved = false;
        this.$inputs = null;
        this.partialCheck = false;
        this.populateInputs();
        var k = $(h);
        if (k.findParamInClass("rule_visible")) {
            var i = k.find(".date_picker");
            if (i.length&&!i.data("built")) {
                var j = this, a = function() {
                    j.initVisibility();
                    PubSub.unsubscribe(i.data("name") + ":built", a)
                };
                PubSub.subscribe(i.data("name") + ":built", a)
            } else {
                this.initVisibility()
            }
        }
        if (l.liveCheck) {
            this.liveCheckInit()
        }
    };
    b.prototype = {
        notify: function(d) {
            var a = this;
            $.each(this.observers, function() {
                this.update(a, d)
            })
        },
        register: function(a) {
            this.observers.push(a)
        },
        liveCheckInit: function() {
            var a = this;
            this.$inputs.each(function() {
                var j = this, g = $(this);
                g.on("blur", function() {
                    if (!$.browser.msie) {
                        g.val($.trim(g.val()))
                    }
                    if (!/^\s*$/.test(g.val())) {
                        DM_Widget_Form.focus_id = null;
                        a.validate()
                    }
                });
                g.on("change", function() {
                    a.validate()
                });
                g.on("focus", function() {
                    DM_Widget_Form.focus_id = j.id
                });
                if (/rule_ajaxchecker/.test(j.className)) {
                    var h = j.id + "_checktimeout", i = function() {
                        DM_Form_Validator.ajaxchecker(a)
                    };
                    g.on("keyup", function(d) {
                        d = d || window.event;
                        var c = d.keyCode ? d.keyCode: d.which;
                        if (Keyboard.is_an_input_modifier_key(c)) {
                            DM_Widget_Form[h] && clearTimeout(DM_Widget_Form[h]);
                            DM_Widget_Form[h] = setTimeout(i, g.findParamInClass("interval") || 500)
                        }
                    });
                    g.on("paste", function() {
                        DM_Widget_Form[h] = setTimeout(i, 100)
                    })
                }
            })
        },
        populateInputs: function() {
            this.$inputs = $(this._metaInputElement).find("input, select, textarea")
        },
        initVisibility: function() {
            var A = $(this._metaInputElement), z = A.findParamsInClass("rule_visible");
            try {
                var t = /(.*)\-(.*)/;
                for (var w = 0; w < z.length; w++) {
                    var i, j, s, e;
                    s = t.exec(z[w]);
                    if (s) {
                        var v = s[1], B = v.split("?");
                        if (B.length > 1&&!A.hasClassName("visible_and")) {
                            this.partialCheck = true
                        }
                        j = s[2];
                        e = j.split("?");
                        if (e) {
                            for (var x = 0; x < B.length; x++) {
                                var y;
                                if (B[x] == this.name) {
                                    y = this
                                } else {
                                    y = DM_Form_Inputs.getInstanceByFieldName(this.dm_form, B[x])
                                }
                                var a = y.getName();
                                this.observedInputs[a] = {};
                                this.observedInputs[a].values = e;
                                this.observedInputs[a].visible = false;
                                y.observeChange();
                                y.register(this);
                                y.notify()
                            }
                        }
                    }
                }
            } catch (u) {}
        },
        observeChange: function() {
            if (!this.changeObserved) {
                var a = this;
                this.$inputs.each(function() {
                    var e = $(this), f = e.attr("type");
                    if (f && f.match(/radio|checkbox/)) {
                        e.on("click", function() {
                            a.notify()
                        })
                    } else {
                        e.on("change", function() {
                            a.notify()
                        })
                    }
                });
                this.changeObserved = true
            }
        },
        update: function(k, h) {
            var i = true, a = k.getValue(), l = this;
            this.observedInputs[k.getName()].visible = false;
            $.each(a, function() {
                if (l.observedInputs[k.getName()].values.indexOf(this + "") >= 0) {
                    l.observedInputs[k.getName()].visible = true
                }
            });
            if (this.partialCheck) {
                i = false;
                $.each(this.observedInputs, function() {
                    i = i || this.visible
                })
            } else {
                $.each(this.observedInputs, function() {
                    i = i && this.visible
                })
            }
            var j = $(this.getContainer());
            if (i) {
                this.visible = true;
                if (j[0]) {
                    j.show();
                    PubSub.publish("form:" + this.dm_form.formName + ":rule:visible", j, true)
                }
            } else {
                this.visible = false;
                if (j[0]) {
                    j.hide();
                    PubSub.publish("form:" + this.dm_form.formName + ":rule:visible", j, false)
                }
            }
            return 
        },
        getContainer: function() {
            var d = this.getRealInput(), a = $(d);
            return ((d && $("#" + d.id + "_cont")[0]) || (d && a.closest(".form_item")[0]) || d)
        },
        validate: function() {
            var o = this, a = true, p = true, m = this.$inputs.filter(function() {
                return $(this).hasClass("check")
            })[0] || this.$inputs.get( - 1), n = $(m);
            var q = $(this._metaInputElement).find(".date_picker");
            if (q[0]) {
                m = $('[name="' + q.data("name") + '"]')[0]
            }
            var l = $.grep(m.className.split(" "), function(c) {
                return c.match(/^rule_/)
            }), t = $.grep(m.className.split(" "), function(c) {
                return c.match(/^hint_/)
            }), r = $.grep(l, function(c) {
                return c.match(/^rule_visible/)
            }).length, s = true;
            if (r) {
                s = this.visible
            }
            if (s) {
                $.each(l, function() {
                    var h = this.replace("rule_", "");
                    if (a !== true) {
                        o.removeErrorForRule(h);
                        return 
                    }
                    var i = true, f;
                    if ($(m).is("select.channel_select")) {
                        b.debug = 1
                    }
                    try {
                        f = DM_Form_Validator.getRuleName(h);
                        var d = DM_Form_Validator.getRuleParams(h);
                        if (d) {
                            if (!f.match(/^visible/)) {
                                a = DM_Form_Validator[f](o.getFilteredValue(), d);
                                if (!a && h.match(/^enume/)) {
                                    i = false
                                }
                            }
                        } else {
                            a = h.match(/^altmultichoice/) ? DM_Form_Validator.altmultichoice(o.getRealInput()) : a = DM_Form_Validator[f](o.getFilteredValue())
                        }
                    } catch (e) {}
                    if (!a && DM_Form_Validator[o.getFormName()]) {
                        if (i) {
                            var c = DM_Form_Validator.getMsgHashIdForRule(h);
                            try {
                                var g = o.getRealInput();
                                DM_Widget_Form.removeInputMsgs(g, "info", "for_ajaxchecker");
                                o.displayError(DM_Form_Validator[o.getFormName()][g.name.replace(/\[.*\]/, "")][f][c], c)
                            } catch (e) {}
                        }
                    } else {
                        o.removeErrorForRule(h)
                    }
                })
            }
            return a
        },
        removeErrorForRule: function(f) {
            var a = DM_Form_Validator.getMsgHashIdForRule(f);
            try {
                DM_Widget_Form.removeInputMsgs(this.getRealInput(), "error", a)
            } catch (e) {}
        },
        getFormName: function() {
            return this.dm_form.getFormName()
        },
        getName: function() {
            return this.getRealInput().name
        },
        getValue: function() {
            var g = this.$inputs.length, a = [];
            if (this.$inputs[0].type.match(/radio|checkbox/)) {
                this.$inputs.each(function() {
                    if (this.checked) {
                        a.push(this.value)
                    }
                })
            } else {
                if (g == 1) {
                    if (this.$inputs[0].type.match(/radio|checkbox/)) {
                        if (this.$inputs[0].checked) {
                            a.push(this.$inputs[0].value)
                        }
                    } else {
                        if (this.$inputs[0].tagName.toLowerCase().match(/input|textarea/)) {
                            a.push(this.$inputs[0].value)
                        } else {
                            if (this.$inputs[0].tagName.toLowerCase() == "select") {
                                a.push(this.$inputs[0].options[this.$inputs[0].selectedIndex].value)
                            }
                        }
                    }
                } else {
                    var h = this.getRealInput(), f = h.value;
                    if (DM_Widget_Form.dateInputSuffixRe && DM_Widget_Form.dateInputSuffixRe.test(h.name)&&!$('[name="' + h.name.replace(DM_Widget_Form.dateInputSuffixRe, TIME_HIDDEN_INPUT_SUFFIX) + '"]').val()) {
                        f = ""
                    }
                    a.push(f)
                }
            }
            return a
        },
        reset: function() {
            delete (this.value);
            this.filteredValue = []
        },
        getFilteredValue: function() {
            var a = [], e = $.grep(this.getRealInput().className.split(" "), function(c) {
                return c.match(/^filter_/)
            }), f = this.getValue();
            $.each(f, function() {
                var c = this;
                $.each(e, function(j, i) {
                    i = i.replace("filter_", "");
                    try {
                        c = DM_Form_Filter[i](c.replace("'", "'"))
                    } catch (d) {}
                });
                a.push(c)
            });
            return a
        },
        getRealInput: function() {
            var a = this.$inputs.get( - 1);
            $realInput = $(a);
            if ($realInput.hasClass("dmco_form_olddate_select")) {
                a = this.inputs[0]
            } else {
                var d = $(this._metaInputElement).find(".date_picker");
                if (d[0]) {
                    a = $('[name="' + d.data("name") + '"]')[0]
                }
            }
            return a
        },
        getInputs: function() {
            return this.$inputs
        },
        displayError: function(k, n) {
            var a = this.getRealInput(), m = $(a).next("span.dmco_form_input_info")[0];
            if (m) {
                a = m
            }
            var i = $(a), j = $(a).closest(".form_input");
            if (j[0]) {
                if (i.hasClass("edit_in_place")) {
                    DM_Widget_Page.notify(k, "error")
                } else {
                    var l = false;
                    n = n ? "msg_" + n : n;
                    if (n) {
                        j.find("div.error_msg").each(function() {
                            var c = $(this);
                            if (c.hasClass(n)) {
                                c.html(k);
                                l = true;
                                return false
                            }
                        })
                    }
                    if (!l) {
                        if (window.ajax_errors_position == "top") {
                            j.addClass("has_error").prepend($('<div class="error_msg ' + (n || "") + '">').html(k))
                        } else {
                            j.addClass("has_error").append($('<div class="error_msg ' + (n || "") + '">').html(k))
                        }
                    }
                }
            }
        },
        displayInfo: function(k, n) {
            var a = this.getRealInput(), m = $(a).next("span.dmco_form_input_info")[0];
            if (m) {
                a = m
            }
            var i = $(a), j = i.closest(".form_input");
            if (j[0]) {
                var l = false;
                n = n ? "msg_" + n : n;
                if (n) {
                    j.find("div.info_msg").each(function() {
                        var c = $(this);
                        if (c.hasClass(n)) {
                            c.html(k);
                            l = true;
                            return false
                        }
                    })
                }
                if (!l) {
                    j.append($('<div class="info_msg ' + (n || "") + '">').html(k))
                }
            }
        }
    };
    return b
}, true);
_DM.define("DM_Form_Filter", function() {
    var b = {
        striptags: function(a) {
            return $.map(a, function(d) {
                if ($.isString(d)) {
                    return $.stripTags(d)
                }
            })
        }
    };
    return b
}, true);
_DM.define("DM_Form_Inputs", function() {
    var b = {
        inputs: {},
        getInstance: function(h, f) {
            var g = h.getFormName();
            var a = null;
            if ($(f).children(".dmco_form_input").length) {
                if ($(f).children().hasClass("multi-checkbox") && $(f).children().length == 1) {
                    a = $(f).find("input.check")[0].id
                } else {
                    a = $(f).children(".dmco_form_input")[0].id
                }
            }
            if (!b.inputs[g]) {
                b.inputs[g] = {}
            }
            if (!b.inputs[g][a]) {
                b.inputs[g][a] = new DM_Form_Input(h, f)
            }
            return b.inputs[g][a]
        },
        reset: function(a) {
            var d = a.getFormName();
            if (!b.inputs[d]) {
                return 
            }
            delete (b.inputs[d])
        },
        getInstanceByFieldName: function(d, a) {
            return b.getInstance(d, d.getInputByName(a))
        }
    };
    return b
}, true);
_DM.define("DM_Form", function() {
    var b = function(f) {
        this.form = f;
        var h = $(f), g = this, a = h.find('input[name="form_name"]')[0];
        this.formName = a ? a.value : "none";
        if (h.hasClass("live_check")) {
            this.liveCheck = true
        }
        this.fieldsToCheck = [];
        this.fields = [];
        h.find(".form_input").each(function() {
            g.fields.push($(this).closest(".form_item")[0])
        });
        if (this.fields.length) {
            h.on("submit", $.proxy(this.validate, this)).find(".form_input.check").each(function() {
                g.fieldsToCheck.push(DM_Form_Inputs.getInstance(g, this))
            })
        }
    };
    b.prototype = {
        validate: function(k) {
            var j = true, a = $(this.form);
            try {
                $.each(this.fieldsToCheck, function() {
                    this.reset();
                    j = this.validate() && j
                })
            } catch (e) {}
            if (!j) {
                k.preventDefault();
                $noticeBox = $("#notice_box", a);
                var l = this.getFormName();
                if ($noticeBox[0] && DM_Form_Validator[l] && DM_Form_Validator[l].error_msg) {
                    var i = $('<div class="dmpi_notice dmpi_notice_error">', a).hide();
                    i.html('<div id="js-error" class="mrg-btm-xl mo_alert media pdg-md brd-rad-md alert-error pdg-end-md alert-md"><div class="media-img mrg-end-md font-md" data-icon=""></div><div class="media-block pdg-top-xs"><div class="pull-end mrg-start-md js-alert-close clickable" title=""><span class="icon-close"></span></div><div class="pdg-end-lg font-md">' + DM_Form_Validator[l].error_msg + "</div></div></div>");
                    $noticeBox.html(i);
                    i.show("fast")
                }
            } else {
                try {
                    if (!a.hasClass("persistent")) {
                        DM_Widget_Form.removeForm(this.form)
                    }
                    if (a.data("alreadyObserved")) {
                        k.preventDefault();
                        if (!a.hasClass("no_submit")&&!this.form.cancelImmediateSubmit) {
                            DM_Ajax.submitForm(this.form)
                        }
                        this.form.cancelImmediateSubmit = false
                    } else {
                        return true
                    }
                } catch (e) {}
            }
            k.preventDefault();
            return false
        },
        getFormName: function() {
            return this.formName
        },
        getInputByName: function(d) {
            var a = $.grep(this.fields, function(c) {
                var f = $(c).find("#" + d)[0];
                if (f) {
                    return true
                }
                return false
            });
            return a[0] && $(a[0]).find(".form_input")[0]
        }
    };
    return b
}, true);
_DM.define("DM_Form_Validator", function() {
    var _self = {
        error_msg: [],
        getRuleParamsPosition: function(fullRule) {
            return fullRule.indexOf("-")
        },
        ruleHasParams: function(fullRule) {
            return this.getRuleParamsPosition(fullRule)>-1
        },
        getRuleName: function(fullRule) {
            if (this.ruleHasParams(fullRule)) {
                return fullRule.slice(0, this.getRuleParamsPosition(fullRule))
            }
            return fullRule
        },
        getRuleParams: function(fullRule) {
            if (this.ruleHasParams(fullRule)) {
                return fullRule.slice(this.getRuleParamsPosition(fullRule) + 1)
            }
            return ""
        },
        getMsgHashIdForRule: function(fullRule) {
            var ruleName = _self.getRuleName(fullRule), ruleParams = _self.getRuleParams(fullRule);
            var msgHashId = str2hex(ruleName);
            if (ruleParams && ["character", "regex", "ajaxchecker"].indexOf(ruleName)!=-1) {
                msgHashId = ruleParams
            }
            return msgHashId
        },
        required: function(values) {
            if (values.length > 0) {
                var filledValues = $.map(values, function(value) {
                    if (!/^\s*$/.test(unescape(value))) {
                        return value
                    }
                });
                if (filledValues.length > 0) {
                    return true
                }
            }
            return false
        },
        altmultichoice: function(input) {
            var realInputValues = [], $input = $(input);
            $input.find("option").each(function() {
                realInputValues.push(unescape($(this).val()))
            });
            $input.val(realInputValues);
            return true
        },
        compare: function(value, params) {
            var $el = params && $("#" + params);
            if (!$el[0]) {
                return false
            }
            value = unescape(value);
            if (value !== $el.val()) {
                return false
            }
            return true
        },
        emailaddress: function(value) {
            return /^([^@]+)@([^@\.]+\.[^@]+)$/.test(unescape(value))
        },
        stringlength: function(values, params) {
            if (values.length > 0) {
                var filledValues = $.map(values, function(value) {
                    value = unescape(value);
                    var result = true;
                    params = params.split("-");
                    if (value) {
                        for (var i = 0; i < params.length; i++) {
                            var test = params[i].split(":");
                            if (test[0] == "max") {
                                result = (value.length <= parseInt(test[1], 10)) && result
                            } else {
                                if (test[0] == "min") {
                                    result = (value.length >= parseInt(test[1], 10)) && result
                                } else {
                                    if (test[0] == "equal") {
                                        result = (value.length == parseInt(test[1], 10)) && result
                                    } else {}
                                }
                            }
                        }
                    }
                    return result
                });
                if (filledValues.length > 0) {
                    return true
                }
            }
            return false
        },
        character: function(value, params) {
            value = $.trim(unescape(value));
            if (!/^\s*$/.test(value)&&!/^\s*$/.test(params)) {
                params = hex2str(params).split("::");
                var pattern = "^[" + params[1] + "]+$";
                if (params[0] == "refuse") {
                    pattern = "^[^" + params[1] + "]+$"
                }
                return value.match(pattern)
            }
            return true
        },
        regex: function(value, params) {
            value = $.trim(unescape(value));
            params = $.parseJSON(params);
            var regex = params.regex, negated = params.negated;
            if (!/^\s*$/.test(value)&&!/^\s*$/.test(params)) {
                params = hex2str(params).slice(1, - 1);
                var regexp = new RegExp(params), result = regexp.test(value);
                return negated?!result : result
            }
            return true
        },
        ajaxchecker: function(dm_form_input) {
            if (!(dm_form_input instanceof DM_Form_Input)) {
                return true
            }
            var input = dm_form_input.getRealInput(), ruleParamsMatches = input.className.match(/rule_ajaxchecker-([a-f0-9]+)/), ruleParams = ruleParamsMatches ? ruleParamsMatches[1]: "", optionsProviderString = hex2str(ruleParams), options = eval(optionsProviderString), defaultOptions = {
                url: null,
                pre_validate: function() {
                    return true
                }
            };
            options = $.extend(defaultOptions, $.isString(options) ? {} : options);
            if (options.url && options.pre_validate(input.id) && dm_form_input.validate()) {
                if (options.url.match(/^\/ajax\//)) {
                    ajax_call(options.url.replace(/\/ajax\//, ""), "check_input", input.id, $(input).val(), function(res) {
                        var jsonRes = $.parseJSON(res);
                        if (jsonRes.valid) {
                            if (jsonRes.msg) {
                                DM_Widget_Form.removeInputMsgs(input, "error");
                                DM_Widget_Form.removeInputMsgs(input, "error", "for_ajaxchecker");
                                dm_form_input.displayInfo.call(dm_form_input, jsonRes.msg, "for_ajaxchecker")
                            }
                        } else {
                            var msg = jsonRes.msg || _self[dm_form_input.getFormName()][input.name.sub(/\[.*\]/, "")].ajaxchecker[ruleParams];
                            if (msg) {
                                DM_Widget_Form.removeInputMsgs(input, "info", "for_ajaxchecker");
                                dm_form_input.displayError.call(dm_form_input, msg, "for_ajaxchecker")
                            }
                        }
                    })
                } else {}
            } else {
                DM_Widget_Form.removeInputMsgs(input, "info", "for_ajaxchecker");
                DM_Widget_Form.removeInputMsgs(input, "error", "for_ajaxchecker");
                if (DM_Widget_Form.focus_id == input.id&&!options.pre_validate(input.id)) {
                    DM_Widget_Form.removeInputMsgs(input, "error")
                }
            }
            return true
        },
        callback: function() {
            return true
        }
    };
    return _self
}, true);
_DM.define("DM_Select2", function() {
    var b = {
        __onDomReady: function() {
            jQuery.fn.scrollTo = function(a, f, e) {
                e = e || null;
                $(this).stop().animate({
                    scrollTop: $(this).scrollTop() - $(this).offset().top + $(a).offset().top
                }, f === undefined ? 1000 : f, e);
                return this
            };
            b.initEvents()
        },
        initEvents: function() {
            b.onClickCalendars();
            b.initQuickSearch()
        },
        onClickCalendars: function() {
            $("body").on("click", ".merged-elements .icon-calendar, .merged-elements .icon-time", function(e) {
                e.preventDefault();
                e.stopImmediatePropagation();
                var g = $(this).closest(".merged-elements");
                if (!g.length) {
                    return 
                }
                var h = g.find(".date_picker,.datepicker,.time_picker").pickadate("picker");
                if (!h) {
                    return 
                }
                var a = h.get("open");
                if (a === true) {
                    return 
                }
                h.open()
            })
        },
        initPicker: function(a) {
            a.not(function() {
                return $(this).data("pickadate") || $(this).hasClass(".no-auto-init-select")
            }).initPicker().pickadate("picker")
        },
        initQuickSearch: function() {
            var a = "";
            var d = null;
            $("body").on("keypress", function(c) {
                if ($searchSelect) {
                    clearTimeout(d);
                    d = setTimeout(function() {
                        a = ""
                    }, 300);
                    var j = c.which || c.keyCode;
                    var e = String.fromCharCode(j);
                    if (/[a-z0-9]/i.test(e)) {
                        a += e.toLowerCase();
                        var i = new RegExp("^" + a + ".*", "gi");
                        $(".select2-result-label", ".select2-drop-active:visible").each(function() {
                            if ($(this).first().text().toLowerCase().match(i)) {
                                $(".select2-results li.select2-highlighted", ".select2-drop-active:visible").removeClass("select2-highlighted");
                                var f = $(this).closest("li").addClass("select2-highlighted");
                                $(".select2-results", ".select2-drop-active:visible").scrollTo(f, 200);
                                return false
                            }
                        })
                    }
                }
            })
        },
        initConcernedFields: function(d, a) {
            d.select2(a).on("select2-open", function() {
                $searchSelect = $(this)
            }).on("select2-close", function() {
                $searchSelect = null
            })
        },
        initSelects: function() {
            if ($(".js-datepicker").length) {
                if (typeof $().initPicker != "undefined") {
                    b.initPicker($(".js-datepicker"))
                } else {
                    $.loadPickadate(DM_Context.language, function() {
                        b.initPicker($(".js-datepicker"))
                    })
                }
            }
            $searchSelect = null;
            var d = $(".form_input select,.form_item select,.selectors select, select.select2").not(function() {
                return $(this).data("select2") || $(this).hasClass("no-auto-init-select") || $(this).closest(".no-auto-init-select-children").length
            });
            if (d.length) {
                var a = {
                    dropdownAutoWidth: true,
                    minimumResultsForSearch: - 1
                };
                if (typeof $().select2 != "undefined") {
                    b.initConcernedFields(d, a)
                } else {
                    DM_WidgetV3.addCss("/css/lib/bower/select2/select2.css");
                    DM_JS.load("/js/lib/bower/select2/select2.js", {
                        name: "Select2",
                        value: $.fn.select2
                    }, function() {
                        b.initConcernedFields(d, a)
                    })
                }
            }
        }
    };
    return b
}, true);
_DM.define("DM_Widget_Form", function() {
    var b = {
        forms: {},
        dateInputSuffixRe: null,
        initEvents: function() {
            DM_Select2.initSelects();
            $("form.dmco_form").each(function() {
                var j = $(this), g = j.find('input[name="form_name"]')[0], i = g ? g.value: null;
                if (j.data("initdone")) {
                    return 
                }
                j.data("initdone", true);
                if (i) {
                    if (!b.forms[i]) {
                        b.forms[i] = new DM_Form(this)
                    }
                    if (DM_Form_Utils.applyPersistentData(j)) {
                        j.find('input[type="text"], textarea').get(0).focus()
                    }
                }
                if (j.hasClass("ajax")) {
                    DM_Ajax.addAjaxForm(this)
                }
                if (j.hasClass("form_with_confirm")) {
                    j.on("submit", function() {
                        return DM_Form[i] && confirm(DM_Form[i].confirm)
                    })
                }
                j.find(".link_to_post a").on("click", function(c) {
                    c.preventDefault();
                    j.submit()
                });
                j.find("textarea.counter, input.counter").each(function() {
                    var d = $(this);
                    if (!d.next().hasClass("dmco_form_input_info")) {
                        var c = d.findParamInClass("count"), e = function(f) {
                            return f + "/" + c + " " + DM_Form.charactersUsedLabel
                        };
                        d.after($('<span class="dmco_form_input_info">')).next().html(e(this.value.length)).prev().on("keyup", function() {
                            var l = $(this), f = l.next();
                            if (this.value.length > c) {
                                f.addClass("error_msg")
                            } else {
                                f.removeClass("error_msg")
                            }
                            f.html(e(this.value.length))
                        })
                    }
                });
                var h = "placeholder" in document.createElement("input");
                if (!h) {
                    $("input, textarea").placeholder()
                }
                j.find("input[data-focus]").first().focus();
                if ($.browser.msie && $("#popupContent").is(":visible")) {
                    $("#popupContent").focus()
                }
                j.find("a#cantread").each(function() {
                    $(this).on("click", function() {
                        $("#captcha_image").html('<div id="captcha_image" style="background-image:url(/dmcaptcha?rand=' + new Date().getTime() + ');"></div>')
                    })
                });
                j.find("input.switcher").each(function() {
                    var c = $(this);
                    c.on("click", function() {
                        c.next().val(c.next().val() == "show" ? "hide" : "show");
                        DM_Form_Inputs.getInstance(b.forms[i], c.parent().parent()[0]).notify()
                    })
                });
                j.find("input.altmultichoice_action").on("click", function() {
                    var d = $(this), f = d.findParamInClass("actionType"), e = hex2str(d.findParamInClass("actionTarget")), c = null, l = null;
                    switch (f) {
                    case"add":
                        c = $("#src_" + e);
                        l = $("#dst_" + e);
                        break;
                    case"remove":
                        c = $("#dst_" + e);
                        l = $("#src_" + e);
                        break
                    }
                    if (c.length > 0 && c.find("option").length > 0 && c.val()) {
                        $(c.val()).each(function(n, k) {
                            l.prepend(c.find('option[value="' + k + '"]').first().remove());
                            l.trigger("change");
                            c.trigger("change")
                        })
                    }
                });
                j.find("div.dmco_form_olddate select").each(function() {
                    var c = $(this), d = c.closest(".form_input"), n = d.find("select.year"), f = d.find("select.month"), m = d.find("select.day"), e = function() {
                        if (n.val() !== 0 && f.val() !== 0 && m.val() !== 0) {
                            var k = new Date();
                            k.setFullYear(parseInt(n.val(), 10), parseInt(f.val() - 1, 10), parseInt(m.val(), 10));
                            c.siblings('input[type="hidden"]').val(k.toDBFormattedString())
                        } else {
                            c.next('input[type="hidden"]').val("")
                        }
                    };
                    e();
                    c.on("change", e)
                });
                var a = j.find("div.dmco_form_datev2 .date_picker, div.dmco_form_datev2 .time_picker");
                if (a[0]) {
                    b.dateInputSuffixRe = window.DATE_HIDDEN_INPUT_SUFFIX ? new RegExp(DATE_HIDDEN_INPUT_SUFFIX) : null;
                    if (!$.loadPickadate) {
                        dbg("Cannot build pickadate date/time pickers. Missing js/lib/dm/plugins/jquery.pickadate/loader.js.");
                        return 
                    }
                    $.loadPickadate(a.data("locale"), function() {
                        a.initPicker()
                    })
                }
                j.find("input.dmco_form_submit").on("click", function() {
                    $(this).closest("form").find("div.ss_msg").each(function() {
                        var d = $(this), c = d.closest("div.form_input");
                        c && c.removeClass("has_error");
                        d.remove()
                    })
                })
            })
        },
        removeInputMsgs: function(f, e, a) {
            a = a&&!a.match(/^msg_/) ? "msg_" + a : a;
            $(f.nodeName.toUpperCase() == "FORM" ? f : $(f).closest(".form_input")[0]).find("div." + e + "_msg").each(function() {
                var c = $(this);
                if ((!a&&!c.hasClass("msg_for_ajaxchecker")) || (a && c.hasClass(a))) {
                    if (e == "error") {
                        c.closest("div.form_input").removeClass("has_error")
                    }
                    c.remove()
                }
            })
        },
        removeFormByName: function(a) {
            var e = b.forms[a];
            if (!e) {
                return 
            }
            var f = e.form;
            $(f).off();
            DM_Form_Inputs.reset(e);
            PubSub.unsubscribe("form:" + a + ":rule:visible");
            delete (b.forms[a])
        },
        removeForm: function(e) {
            var f = $(e), a = f.find('input[name="form_name"]')[0];
            if (!a) {
                return 
            }
            b.removeFormByName(a.value)
        },
        __onDomReady: function() {
            b.initEvents();
            PubSub.subscribe("forms:refresh_init", b.initEvents, this);
            PubSub.subscribe("popup:opened", b.initEvents, this)
        }
    };
    return b
}, true);
_DM.define("SfForm_Constraints", function(l) {
    var i = "submit", n = /([\-]+(.))/g, m = /^data-/, k = function(a) {
        return a.replace(n, function(e, c, d, b) {
            return b ? d.toUpperCase() : d
        })
    }, o = function(a) {
        return a != l
    }, p = function(a) {
        return a.replace(m, "")
    };
    var j = {
        isConstraintAttribute: function(a) {
            return a.match(m)&&!!j.validators[k(a.replace("data-", ""))]
        },
        initInputConstraints: function(b) {
            var a = [];
            $(b.attributes).each(function() {
                if (!this.specified) {
                    return 
                }
                if (j.isConstraintAttribute(this.name)) {
                    var c = this.value[0] == "{" ? $.parseJSON(this.value): this.value, d = {
                        name: p(this.name),
                        params: c,
                        valid: true,
                        checkon: c.checkon
                    };
                    if (o(c.debounce)) {
                        d.debounce = c.debounce
                    }
                    if (this.name == "required") {
                        a = [d].concat(a)
                    } else {
                        a.push(d)
                    }
                }
            });
            b.constraints = a
        },
        resetInputConstraints: function(c, d, e) {
            for (var b in c.constraints) {
                var a = c.constraints[b];
                if ((!d || a.checkon == d) && a.name != e) {
                    a.valid = true;
                    a.freezeDebounce = true
                }
            }
        },
        checkInputConstraint: function(d, a) {
            var e = j.validators[a.name](d.value, a.params), b = e, c;
            if (typeof e !== "boolean") {
                b = false;
                c = a.params[e].message
            } else {
                if (!e) {
                    c = a.params.message
                }
            }
            a.valid = b;
            a.errorMessage = b ? "" : c
        },
        checkAllInputConstraints: function(a, c) {
            var h = c ? (c == i): true, d = true, t = {}, b = function(q) {
                return function() {
                    if (q.freezeDebounce) {
                        return 
                    }
                    j.checkInputConstraint(a, q);
                    SfForm.updateInputErrors(a)
                }
            };
            for (var e = 0, f = a.constraints.length; e < f; e++) {
                var g = a.constraints[e];
                if (!c || (g.checkon && g.checkon.match(c)) || h) {
                    if (g.debounce) {
                        g.freezeDebounce = false;
                        if (!g.debouncedHandler) {
                            g.debouncedHandler = debounce(b(g), g.debounce)
                        }
                        g.debouncedHandler()
                    } else {
                        j.checkInputConstraint(a, g)
                    }
                }
                t = g;
                d = d && g.valid;
                if (a.required && t.name == "required"&&!t.valid) {
                    j.resetInputConstraints(a, null, "required");
                    break
                }
            }
            a.isValid = d
        },
        validators: {
            required: function(a) {
                return o(a)&&!$.blank(a)
            },
            length: function(a, b) {
                var c = true, d;
                if (b.min) {
                    c = a.length >= b.min.limit;
                    if (!c) {
                        d = "min"
                    }
                }
                if (c && b.max) {
                    c = a.length < b.max.limit;
                    if (!c) {
                        d = "max"
                    }
                }
                return d || c
            }
        }
    };
    return j
}, true);
_DM.define("SfForm", function() {
    var v = "with-validation", o = ".js-form_error", p = ".js-form-error-template", n = ".js-error-message-content", x = "need-validation", u = ".js-form_field", w = ".js-form_input", y = ".js-input_errors", z = "has_error", q = ".js-input-error-template", t = ".js-input_info .count", r = "input-length-help", s = {
        initEvents: function() {
            $(document).on("submit", function(a) {
                if (!a.target[v]) {
                    return 
                }
                var b = $(a.target);
                if (!s.isFormValid(b)) {
                    s.updateFormErrors(b);
                    a.preventDefault()
                } else {
                    s.cleanFormErrors(b)
                }
            }).on("blur", "input, textarea", function(a) {
                if (!a.target[x]) {
                    return 
                }
                s.validateInput(a.target, "blur")
            }).on("focus", "input, textarea", function(a) {
                if (!a.target[x]) {
                    return 
                }
                SfForm_Constraints.resetInputConstraints(a.target, "blur")
            }).on("keyup", "input, textarea", function(a) {
                if (!a.target[x]) {
                    return 
                }
                s.validateInput(a.target, "keyup");
                if (a.target.dataSubscribers) {
                    for (var b in a.target.dataSubscribers) {
                        a.target.dataSubscribers[b]()
                    }
                }
            })
        },
        initInputDataSubscribers: function(c) {
            if ($(c).data(r)) {
                c.dataSubscribers = c.dataSubscribers || [];
                var a = $(c).closest(w).find(t), b = function() {
                    a.html(c.value.length)
                };
                c.dataSubscribers.push(b);
                b()
            }
        },
        updateFormErrors: function(d) {
            var b = true;
            s.getValidableInputs(d).each(function() {
                if (!this.isValid) {
                    b = false;
                    s.updateInputErrors(this)
                } else {
                    s.cleanInputErrors(this)
                }
            });
            var a = $(d), c = a.find(o);
            if (!b) {
                d.$formErrorTemplate = d.$formErrorTemplate || $($.trim(a.find(p).html()));
                if (!c.length) {
                    a.prepend(d.$formErrorTemplate)
                } else {
                    c.replaceWith(d.$formErrorTemplate);
                    c.show()
                }
            } else {
                c.hide()
            }
        },
        cleanFormErrors: function(b) {
            var a = $(b);
            a.find(o).remove();
            s.getValidableInputs(b).each(function() {
                s.cleanInputErrors(this)
            })
        },
        isFormValid: function(b) {
            var a = true;
            s.getValidableInputs(b).each(function() {
                SfForm_Constraints.checkAllInputConstraints(this);
                a = a && this.isValid
            });
            return a
        },
        validateInput: function(a, b) {
            SfForm_Constraints.checkAllInputConstraints(a, b);
            s.updateInputErrors(a)
        },
        cleanInputErrors: function(a) {
            $(a).closest(u).find(y).empty().closest(w).removeClass(z)
        },
        updateInputErrors: function(g) {
            var d = [], a = $(g), c = a.closest(u).find(y), f = a.closest("form")[0].fieldErrorTemplate;
            for (var e in g.constraints) {
                var b = g.constraints[e];
                if (!b.valid) {
                    d.push(f.replace("__TYPE__", b.name).replace("__MESSAGE__", b.errorMessage))
                }
            }
            if (d.length) {
                c.html(d.join("")).closest(w).addClass(z)
            } else {
                c.empty().closest(w).removeClass(z)
            }
        },
        getValidableInputs: function(a) {
            return $(a).find("[" + x + "]")
        },
        handleForm: function(a) {
            if (a[v]) {
                return 
            }
            a[v] = true;
            s.getValidableInputs(a).each(function() {
                if (!this[x]) {
                    this[x] = true;
                    SfForm_Constraints.initInputConstraints(this);
                    s.initInputDataSubscribers(this)
                }
            })
        },
        __onDomReady: function() {
            s.initEvents();
            $("form[" + v + "]").each(function() {
                s.handleForm(this);
                this.fieldErrorTemplate = $.trim($(this).find(q).html())
            })
        }
    };
    return s
}, true);
_DM.define("DM_LocaleSelector", function() {
    var b = {
        initEvents: function() {
            $("#footer").on("click", "#country_selector a", DM_LocaleSelector.onCountryFlagClick)
        },
        onCountryFlagClick: function() {
            var a = $(this).findParamInClass("country");
            var d = $(this).findParamInClass("language");
            b.changeLocale(d, a)
        },
        initCountrySelectorOnLoad: function() {
            var e = $("#country_selector a");
            var f = document.location.search;
            var a = document.location.hash;
            if ($.trim(f) !== "") {
                e.each(function() {
                    $(this).attr("href", $(this).attr("href") + f)
                })
            }
            if ($.trim(a) !== "" && typeof DM_LocaleSelector_routeName != "undefined" && DM_LocaleSelector_routeName == "faq") {
                e.each(function() {
                    $(this).attr("href", $(this).attr("href") + a)
                })
            }
        },
        changeLocale: function(d, a) {
            if (a) {
                d = d + "_" + a.toUpperCase()
            }
            setCookie(DM_LoginCookie, "", - 10);
            setCookie("lang", d, 100)
        },
        __onDomReady: function() {
            setTimeout(function() {
                DM_WidgetV3.addCss("/css/views/shared/flag.css")
            }, 4000);
            b.initEvents();
            if ($("#country_selector").size()) {
                b.initCountrySelectorOnLoad()
            }
        }
    };
    return b
}, true);
_DM.define("DM_BrowsingContext", function() {
    var b = function() {
        this.timeout = 1800;
        this.maxsize = 3000;
        this.list = this.getList();
        if (this.expire()) {
            this.save()
        }
        $($.proxy(this.apply, this))
    };
    b.prototype = {
        apply: function(a) {
            this.initEvents(a)
        },
        initEvents: function(f) {
            var h = "a[data-context]", a = $.isString(f) ? f + " " + h: h, g = this;
            $(a).each(function() {
                if (this.dm_browsingcontext_observed) {
                    return 
                }
                this.dm_browsingcontext_observed = true;
                $(this).on("click", $.proxy(g.put, g, this.href.split("#")[0].split("?")[0], this.getAttribute("data-context")))
            })
        },
        getList: function() {
            var f = {}, a = getCookie("dmc");
            try {
                if (a) {
                    f = $.parseJSON(unescape(a))
                }
                if ($.isArray(f)) {
                    f = {}
                }
            } catch (e) {}
            return f
        },
        put: function(d, a) {
            this.list[this.sign(d)] = {
                c: a,
                t: this.timestamp()
            };
            this.reduce();
            this.save()
        },
        save: function() {
            setCookie("dmc", this.serialize())
        },
        serialize: function() {
            return escape($.toJSONString(this.list))
        },
        expire: function() {
            var f = false;
            for (var a in this.list) {
                if (this.list.hasOwnProperty(a)) {
                    var e = this.list[a].t;
                    if (!$.isNumeric(e) || this.timestamp() > (e + this.timeout)) {
                        delete (this.list[a]);
                        f = true
                    }
                }
            }
            return f
        },
        timestamp: function() {
            return Math.round(new Date().getTime() / 1000)
        },
        reduce: function() {
            while (this.serialize().length > this.maxsize) {
                var a = null;
                for (var d in this.list) {
                    if (this.list.hasOwnProperty(d)) {
                        if (a === null || this.list[d].t < this.list[a].t) {
                            a = d
                        }
                    }
                }
                if (!a) {
                    break
                }
                delete (this.list[a])
            }
        },
        sign: function(a) {
            return this.crc32(a.replace(/[^a-zA-Z0-9_]+/g, ""))
        },
        crc32: function(j, l) {
            var m = "00000000 77073096 EE0E612C 990951BA 076DC419 706AF48F E963A535 9E6495A3 0EDB8832 79DCB8A4 E0D5E91E 97D2D988 09B64C2B 7EB17CBD E7B82D07 90BF1D91 1DB71064 6AB020F2 F3B97148 84BE41DE 1ADAD47D 6DDDE4EB F4D4B551 83D385C7 136C9856 646BA8C0 FD62F97A 8A65C9EC 14015C4F 63066CD9 FA0F3D63 8D080DF5 3B6E20C8 4C69105E D56041E4 A2677172 3C03E4D1 4B04D447 D20D85FD A50AB56B 35B5A8FA 42B2986C DBBBC9D6 ACBCF940 32D86CE3 45DF5C75 DCD60DCF ABD13D59 26D930AC 51DE003A C8D75180 BFD06116 21B4F4B5 56B3C423 CFBA9599 B8BDA50F 2802B89E 5F058808 C60CD9B2 B10BE924 2F6F7C87 58684C11 C1611DAB B6662D3D 76DC4190 01DB7106 98D220BC EFD5102A 71B18589 06B6B51F 9FBFE4A5 E8B8D433 7807C9A2 0F00F934 9609A88E E10E9818 7F6A0DBB 086D3D2D 91646C97 E6635C01 6B6B51F4 1C6C6162 856530D8 F262004E 6C0695ED 1B01A57B 8208F4C1 F50FC457 65B0D9C6 12B7E950 8BBEB8EA FCB9887C 62DD1DDF 15DA2D49 8CD37CF3 FBD44C65 4DB26158 3AB551CE A3BC0074 D4BB30E2 4ADFA541 3DD895D7 A4D1C46D D3D6F4FB 4369E96A 346ED9FC AD678846 DA60B8D0 44042D73 33031DE5 AA0A4C5F DD0D7CC9 5005713C 270241AA BE0B1010 C90C2086 5768B525 206F85B3 B966D409 CE61E49F 5EDEF90E 29D9C998 B0D09822 C7D7A8B4 59B33D17 2EB40D81 B7BD5C3B C0BA6CAD EDB88320 9ABFB3B6 03B6E20C 74B1D29A EAD54739 9DD277AF 04DB2615 73DC1683 E3630B12 94643B84 0D6D6A3E 7A6A5AA8 E40ECF0B 9309FF9D 0A00AE27 7D079EB1 F00F9344 8708A3D2 1E01F268 6906C2FE F762575D 806567CB 196C3671 6E6B06E7 FED41B76 89D32BE0 10DA7A5A 67DD4ACC F9B9DF6F 8EBEEFF9 17B7BE43 60B08ED5 D6D6A3E8 A1D1937E 38D8C2C4 4FDFF252 D1BB67F1 A6BC5767 3FB506DD 48B2364B D80D2BDA AF0A1B4C 36034AF6 41047A60 DF60EFC3 A867DF55 316E8EEF 4669BE79 CB61B38C BC66831A 256FD2A0 5268E236 CC0C7795 BB0B4703 220216B9 5505262F C5BA3BBE B2BD0B28 2BB45A92 5CB36A04 C2D7FFA7 B5D0CF31 2CD99E8B 5BDEAE1D 9B64C2B0 EC63F226 756AA39C 026D930A 9C0906A9 EB0E363F 72076785 05005713 95BF4A82 E2B87A14 7BB12BAE 0CB61B38 92D28E9B E5D5BE0D 7CDCEFB7 0BDBDF21 86D3D2D4 F1D4E242 68DDB3F8 1FDA836E 81BE16CD F6B9265B 6FB077E1 18B74777 88085AE6 FF0F6A70 66063BCA 11010B5C 8F659EFF F862AE69 616BFFD3 166CCF45 A00AE278 D70DD2EE 4E048354 3903B3C2 A7672661 D06016F7 4969474D 3E6E77DB AED16A4A D9D65ADC 40DF0B66 37D83BF0 A9BCAE53 DEBB9EC5 47B2CF7F 30B5FFE9 BDBDF21C CABAC28A 53B39330 24B4A3A6 BAD03605 CDD70693 54DE5729 23D967BF B3667A2E C4614AB8 5D681B02 2A6F2B94 B40BBE37 C30C8EA1 5A05DF1B 2D02EF8D";
            if (l == window.__undefined) {
                l = 0
            }
            var i = 0, a = 0;
            l = l^( - 1);
            for (var n = 0, k = j.length; n < k; n++) {
                i = (l^j.charCodeAt(n)) & 255;
                a = "0x" + m.substr(i * 9, 8);
                l = (l>>>8)^a
            }
            return l^( - 1)
        }
    };
    window.dm_browsingcontext = new b();
    return b
}, true);
_DM.define("Sd_Header", function() {
    var b = {
        canFocusSearch: true,
        initEvents: function() {
            $("#header_search").submit(Sd_Header.onSearchSubmit);
            $("#bodyall").on("click", "#close_super_header", b.onCloseSuperHeaderClick);
            $("#bodyall").on("touchstart click touch", "#close_cookies_policy", b.onCloseCookiesPolicyClick);
            $("#bodyall").on("touchstart click", ".tipsy-profile-menu-inner li", b.onClickMenuLi);
            $("#bodyall").on("click", "#login_info img.clickable", b.onClickAvatar);
            $("#bodyall").on("keyup", "#input_search", b.onKeydownSearchInput)
        },
        onKeydownSearchInput: function(h) {
            var i = $.trim($("#input_search").val()), j = String.fromCharCode(h.keyCode), a = j.match(/\w/), e = (h.keyCode == 8 || h.keyCode == 46);
            if (a || e) {
                $("#input_search_copy").val(i)
            }
        },
        onClickMenuLi: function(a) {
            window.location = $(this).find("a").attr("href");
            $(this).siblings().removeClass("selected").end().addClass("selected");
            a.stopImmediatePropagation();
            return false
        },
        onClickAvatar: function() {
            window.location = $(this).data("href")
        },
        onSearchSubmit: function() {
            var j = $("#input_search");
            var l = new RegExp("^(\\s|\\.|~)*$", "g");
            if (j.length) {
                if (l.test(j.val())) {} else {
                    var o = ((b.universalSearch&&!DM_Context.search_source_type) || Request.getRequestURI().indexOf("relevance/universal/search")!=-1) ? "/universal": "";
                    o = "/relevance" + o + "/search/" + encodeURIComponent(Request.cleanSearch(j.val())).replace(/%20/g, "+") + "/1";
                    var p = true;
                    if (typeof DM_Context.search_source_type != "undefined") {
                        var n = DM_Context.search_source_type;
                        if ($.inArray(n, ["user", "playlist", "group"])!=-1) {
                            o = "/" + n + "s" + o;
                            p = false
                        }
                    }
                    if (p) {
                        o = "/" + $("#search_locale").val() + o
                    }
                    var k = $(".js-suggestion_container.selected"), m = parseInt($(".suggestions div.js-suggestion_container").index(k) + 1, 10), a = "Characters: " + $("#input_search_copy").val().length + ", Spot: " + m + ", String: " + (m > 0 ? $("#input_search_copy").val() : "none");
                    if ($(".suggestions div.js-suggestion_container").length > 0) {
                        setCookie("sc_action_track_auto_suggest", a)
                    }
                    deleteCookie("sc_action_first_click_after_search_done");
                    document.location.href = o
                }
            }
            return false
        },
        focusSearchField: function() {
            if (window.page_has_intersticial ||!b.canFocusSearch) {
                return 
            }
            if (document.body.className.match(/pg_wtw/)) {
                $("#input_search").focus()
            }
        },
        searchFieldFocusOnLoad: function() {
            b.focusSearchField();
            var a = Request.getHashParams();
            if (typeof Sd_Header_search != "undefined" && typeof a.user_search == "undefined") {
                $("#input_search").val(Sd_Header_search)
            }
        },
        lockFocus: function() {
            b.canFocusSearch = false
        },
        unlockFocus: function() {
            b.canFocusSearch = true
        },
        onCloseSuperHeaderClick: function() {
            setCookie("hideSuperHeader", 1, 7);
            $(".sd_metaheader").remove();
            PubSub.publish("dm:close_super_header")
        },
        onCloseCookiesPolicyClick: function() {
            setCookie("hideCookiesPolicy", 1, 365 * 10);
            $(".sd_cookiespolicy").remove();
            PubSub.publish("dm:metaheader:closed")
        },
        webappRedirect: function() {
            this.displayName = "webappRedirect";
            setCookie("force_desktop_version", "", - 1);
            document.location.reload(true)
        },
        hideConnectLinkAndFB: function() {
            if ($(document.body).hasClass("dm_page_html_login")) {
                $("#login_info").children().eq(0).remove();
                $("#login_infos").remove()
            }
        },
        hideAllAfterRegisterStep1: function() {
            $(".dmpi_registerfull").css("width", "auto");
            $(".register_form_box").addClass("register_step2").removeClass("mrg-end-xl").removeClass("pdg-end-xl");
            $(".register_right_box").hide()
        },
        hideAllAfterRegister: function() {
            $("#popupContent div.register_form_box").css("width", "auto");
            $(".register_form_box").addClass("register_step2").removeClass("mrg-end-xl").removeClass("pdg-end-xl");
            $(".register_right_box").hide();
            $("#popupContent div.register_right_box, #popupContent div.box_sep").hide();
            $("form.dmpi_registerform").removeClass("dmpi_registerform");
            if (typeof DM_Widget_Popup != "undefined" && DM_Widget_Popup.instance) {
                DM_Widget_Popup.instance.center()
            }
        },
        hideAllAfterLogin: function() {
            if ($("div.dmpi_loginfull").length) {
                $("div.dmpi_loginfull").css("width", "auto");
                $(".login_right_box, #popup_title, #popupContent div.box_sep").hide();
                $(".login_form_box").addClass("confirm_message").removeClass("mrg-end-xl").removeClass("pdg-end-xl");
                $(".login_form").removeClass("dmpi_login");
                if (typeof DM_Widget_Popup != "undefined" && DM_Widget_Popup.instance) {
                    DM_Widget_Popup.instance.center()
                }
            }
        },
        __onDomReady: function() {
            b.initEvents();
            b.searchFieldFocusOnLoad();
            b.hideConnectLinkAndFB()
        },
        __initialize: function() {
            HeaderLogin.update();
            this.webappRedirect.displayName = "webappRedirect";
            PubSub.subscribe("dm:metaheader:closed", b.onCloseSuperHeaderClick);
            PubSub.subscribe("masscast:intersticial_opened", b.lockFocus);
            PubSub.subscribe("masscast:intersticial_closed", function() {
                b.unlockFocus();
                b.focusSearchField()
            })
        }
    };
    return b
}, true);
_DM.define("Sd_Footer", function() {
    var b = {
        $footer: null,
        scrollTimeout: null,
        addingContent: false,
        showButtonRatio: 3,
        __initialize: function() {
            b.$footer = $("#footer");
            b.$window = $(window);
            b.$document = $(document);
            b.$backToTop = $(".js-back-to-top");
            setTimeout(function() {
                b.showBackToTop();
                if (b.needsBackToTop()) {
                    $(window).on("scroll", b.onScroll)
                }
            }, 3000);
            b.initEvents();
            PubSub.subscribe("footer:unfold", Sd_Footer.onUnfold).subscribe("footer:fold", Sd_Footer.onFold)
        },
        initEvents: function() {
            $(document).on("click", ".js-back-to-top", b.scrollToTop).on("click", ".js-family-filter-toggle", b.showFamilyfilter);
            $(".ff-no-ellipsis").removeAttr("title")
        },
        showFamilyfilter: function() {
            DM_WidgetV3.get("/controller/Page_FamilyFilter", "popup")
        },
        onUnfold: function() {
            b.$footer.addClass("unfolded")
        },
        onFold: function() {
            b.$footer.removeClass("unfolded")
        },
        onScroll: function(a) {
            clearTimeout(b.scrollTimeout);
            b.scrollTimeout = setTimeout(b.showBackToTop, 300)
        },
        needsBackToTop: function() {
            var d = b.$document.height();
            var a = b.$window.height();
            return (d / a) >= b.showButtonRatio
        },
        needsShowButton: function() {
            var a = b.$window.scrollTop();
            var d = b.$window.height();
            return a >= d
        },
        showBackToTop: function() {
            if (b.needsBackToTop() && b.needsShowButton()) {
                b.$backToTop.slideDown("fast")
            } else {
                b.$backToTop.slideUp("fast")
            }
        },
        scrollToTop: function() {
            $("html, body").animate({
                scrollTop: 0
            })
        }
    };
    return b
}, true);
_DM.define("DM_SocialAuth", function() {
    var _providers = {
        facebook: {
            jsObject: "DM_Facebook",
            label: "Facebook"
        },
        googleplus: {
            jsObject: "DM_GooglePlus",
            label: "Google+"
        }
    }, _messages = ["mappedMsg", "unloggedMsg", "unauthorizedMsg", "unlinkConfirmMsg", "wrongMappingMsg", "unlinkWarningMsg"], _isWidgetOnly = false, _dmNeed = false, _fid = null, _urlback = null, _isMm = false, _userLoginData = null, _noReload = false, _isOnTour = false, _version = 1, _self = {
        getLoginParams: function() {
            var params = {
                isWidgetOnly: _isWidgetOnly,
                dmNeed: _dmNeed,
                fid: _fid,
                urlback: _urlback,
                isMm: _isMm
            };
            return params
        },
        connect: function(provider, options) {
            if (typeof options !== "undefined") {
                _isWidgetOnly = options.widget_only || false;
                _dmNeed = options.dm_need || false;
                _fid = options._fid || null;
                _isMm = options.mm || false;
                _urlback = options.urlback || null;
                _noReload = options.noReload || false;
                _callback = options.callback || null
            }
            if (_providers.hasOwnProperty(provider)===-1) {
                console.error("The following provider is not set in DM_SocialAuth: " + _providers[provider].label);
                return 
            } else {
                window[_providers[provider].jsObject].connect()
            }
        },
        afterSocialLogged: function(params) {
            if (!_isOnTour && window.DM_Widget_Popup) {
                DM_Widget_Popup.close()
            }
            _self.mapUserToDaily(params)
        },
        mapUserToDaily: function(params) {
            log("DM_SocialAuth", "Ajax call to map user to Daily base");
            switch (params.provider) {
            case"facebook":
                return ajax_call("social", "authenticate", $.toJSONString({
                    provider: "facebook"
                }), _self.onDailyMapped);
            case"googleplus":
                return ajax_call("social", "authenticate", $.toJSONString({
                    provider: "googleplus",
                    code: params.code
                }), _self.onDailyMapped)
            }
        },
        onDailyMapped: function(response) {
            log("DM_SocialAuth", "On ajax mapping function return");
            log("DM_SocialAuth", response);
            var res = $.parseJSON(decodeURIComponent(response)), data = null;
            if (!res.status) {
                log("DM_SocialAuth", "Cannot map user on Daily: " + res.message);
                console.error("DM_SocialAuth cannot log user to Dailymotion: " + res.message);
                if (res.reason == "WRONG_MAPPING") {
                    PubSub.publish("dm:social:wrong_mapping", res.provider)
                }
            } else {
                PubSub.subscribe("dm:social:cookie:updated", function() {
                    _self.executeCallbackAfterMapping(res)
                });
                switch (res.action) {
                case"register":
                    _self.handleSocialRegistration(res);
                    break;
                case"login":
                    PubSub.subscribe("dm:social:cookie:updated", _self.handleSocialLogin);
                    break;
                case"link":
                case"refresh":
                    PubSub.subscribe("dm:social:cookie:updated", _self.handleSocialRefresh);
                    break
                }
                PubSub.publish("dm:social:mapped", res)
            }
        },
        executeCallbackAfterMapping: function(res) {
            if (typeof _callback == "function") {
                _callback(res)
            } else {
                if (typeof _callback == "string") {
                    var func = eval(_callback);
                    func(res)
                }
            }
        },
        handleSocialRegistration: function(data) {
            log("DM_SocialAuth", "User need to be registered. Call to open register step2 popup.");
            var params = data;
            if (_isWidgetOnly) {
                params.widgetOnly = true;
                PubSub.publish("dm:social:registration_needed", params)
            } else {
                PubSub.publish("dm:social:registration_needed", params)
            }
            _self.logStats("register", data.provider)
        },
        handleSocialLogin: function(provider) {
            PubSub.unsubscribe("dm:social:cookie:updated", _self.handleSocialLogin);
            _self.logStats("login", provider);
            if (_fid) {
                DM_Ajax.completeTransaction(_fid)
            } else {
                if (!_noReload) {
                    return _self.customReload()
                }
            }
            log("DM_SocialAuth", "No page reload because _noReload parameter is set to false.")
        },
        handleSocialRefresh: function() {
            HeaderLogin.update()
        },
        customReload: function() {
            var location = "http://" + document.location.hostname + document.location.pathname + document.location.search;
            if (location.indexOf("fbc=") > 0) {
                document.location = location.replace(/fb_source=(.*)&/, "").replace(/fbc=(\d+)/, "fbc=" + Math.floor(Math.random() * 1000))
            } else {
                var sep = "?";
                if (location.indexOf("?") > 0) {
                    sep = "&"
                }
                var url = location + sep + "fbc=" + Math.floor(Math.random() * 1000);
                document.location = url.replace(/fb_source=(.*)&/, "")
            }
        },
        logStats: function(action, provider) {
            switch (action) {
            case"login":
                DM_Stats.trigger({
                    action: "login",
                    context: provider
                });
                break;
            case"register":
                DM_Stats.trigger({
                    action: "register",
                    context: provider + "-connect"
                });
                break
            }
        },
        unlinkAfterConnect: function(response) {
            if (response.status) {
                log("DM_SocialAuth", "Unlink account after  connect");
                var provider = response.provider, loginCookie = $.parseJSON(getCookie(DM_LoginCookie)), userLogin = loginCookie.login;
                window[_providers[provider].jsObject].unlink(userLogin)
            } else {
                log("DM_SocialAuth", "Cannot map user on Daily: " + res.message)
            }
        },
        getSocialCookie: function() {
            var socialCookie = ($.parseJSON(getCookie("dm_sc")) !== null) ? $.parseJSON(getCookie("dm_sc")): {};
            return socialCookie
        },
        initSocialMsg: function() {
            var new_msg;
            $.each(_messages, function(i, m) {
                $.each(_providers, function(provider, data) {
                    new_msg = _self[m].replace(/#provider#/g, data.label);
                    window[data.jsObject][m] = new_msg
                })
            })
        },
        __onDomReady: function() {
            if ($(".pg_profile").length > 0) {
                _self.initSocialMsg()
            }
        },
        __initialize: function() {
            PubSub.subscribe("dm-tour:start", function() {
                _isOnTour = true
            });
            PubSub.subscribe("dm-tour:end", function() {
                _isOnTour = false
            })
        }
    };
    return _self
}, true);
_DM.define("DM_Facebook", function() {
    var b = {
        onDailyMapped: function(d) {
            if (d.provider != "facebook") {
                return 
            }
            var a = DM_SocialAuth.getSocialCookie();
            a.facebook = d.socialData;
            setCookie("dm_sc", $.toJSONString(a));
            PubSub.publish("dm:social:cookie:updated", "facebook")
        },
        onFacebookResponse: function(a) {
            if (a.authResponse) {
                log("DM_Facebook", 'User logged on FB and linked with "Daily FB Connect App"');
                PubSub.publish("dm:fb:status:logged", {
                    onLoginOrRegister: true,
                    provider: "facebook"
                });
                return DM_SocialAuth.afterSocialLogged({
                    provider: "facebook"
                })
            }
            log("DM_Facebook", "User cancelled login")
        },
        connect: function(a) {
            PubSub.publish("dm:fb:connecting");
            log("DM_Facebook", "try to connect to FB");
            FB.login(b.onFacebookResponse, {
                scope: "email"
            })
        },
        requestWritePermissions: function(a) {
            a = typeof a === "function" ? a : false;
            FB.login(function(d) {
                if (d.status === "connected") {
                    FB.api("me/permissions", function(c) {
                        if ("publish_actions" in c.data[0]) {
                            a && a(true)
                        } else {
                            a && a(false)
                        }
                    })
                } else {
                    a && a(false)
                }
            }, {
                scope: "publish_actions"
            })
        },
        disconnect: function(a) {
            if (a == "unknown") {
                PubSub.publish("dm:fb:status:unlogged", {
                    provider: "facebook"
                })
            } else {
                if (a == "not_authorized") {
                    PubSub.publish("dm:fb:status:not_authorized", {
                        provider: "facebook"
                    })
                }
            }
            var d = DM_SocialAuth.getSocialCookie();
            if (d && typeof d.facebook != "undefined") {
                delete d.facebook
            }
            setCookie("dm_sc", $.toJSONString(d))
        },
        unlink: function(a) {
            PubSub.publish("dm:fb:unlinking");
            b.removePermissions(function() {
                ajax_call("social", "unlinkAccount", $.toJSONString({
                    provider: "facebook",
                    userLogin: a
                }), function(d) {
                    b.disconnect();
                    PubSub.publish("dm:fb:unlinked", d)
                })
            })
        },
        removePermissions: function(a) {
            log("DM_Facebook", "remove Dailymotion permissions on user app settings");
            FB.api("/me/permissions", "DELETE", function(d) {
                if (d) {
                    if (d.error && d.error.type == "OAuthException") {} else {
                        a()
                    }
                }
            })
        },
        handleSessionResponse: function(a) {
            if (!getCookie("sid")) {
                return 
            }
            log("DM_Facebook", a);
            if (typeof a.authResponse == "undefined" ||!a.authResponse) {
                log("DM_Facebook", "User is not logged on Facebook OR has not authorized application");
                return b.disconnect(a.status)
            }
            log("DM_Facebook", "User is logged on Facebook");
            ajax_call("social", "getMappedSocialData", $.toJSONString({
                provider: "facebook"
            }), function(d) {
                if (d) {
                    b.afterMapping(a.authResponse.userID, a.status, d)
                }
            })
        },
        afterMapping: function(f, h, a) {
            var g = $.parseJSON(a);
            if (f != g.id) {
                b.disconnect(h);
                return 
            }
            FB.api("/me/permissions", function(c) {
                if (!c.data[0].email) {
                    log("DM_Facebook", "User has not accept all permissons");
                    b.disconnect("not_authorized");
                    return 
                }
                log("DM_Facebook", "User approves email permission");
                log("DM_Facebook", 'User logged on FB and linked with "Daily FB Connect App"');
                PubSub.publish("dm:fb:status:logged", {
                    onLoginOrRegister: false,
                    provider: "facebook"
                });
                var d = DM_SocialAuth.getSocialCookie();
                if (typeof d.facebook == "undefined") {
                    d.facebook = g;
                    setCookie("dm_sc", $.toJSONString(d));
                    PubSub.publish("dm:social:cookie:updated", "facebook")
                }
            })
        },
        shareToFeed: function(d, a) {
            FB.ui({
                method: "share",
                href: d
            }, function(c) {
                typeof a === "function" && a(c)
            })
        },
        postToFeed: function(h, f, a) {
            var g = {
                message: h,
                "fb:explicitly_shared": true
            };
            if (f) {
                g.link = f
            }
            b.requestWritePermissions(function(c) {
                FB.api("/me/feed", "post", g, function(d) {
                    typeof a === "function" && a(d)
                })
            })
        },
        commentVideo: function(f, e, a) {
            if (DM_Context.env in ["dev", "stage-01"]) {
                console.info("Open Graph actions will only work in prod or preprod. Be sure to use a url from these environments to actually check if this works.")
            }
            e = e || "";
            FB.api("me/dailymotionapp:comment", "post", {
                other: f ? f: window.location.href,
                "fb:explicitly_shared": true,
                message: e
            }, function(c) {
                DM_Stats.trigger({
                    action: "share_new_video_comment_on_facebook"
                });
                typeof a === "function" && a(c)
            })
        },
        init: function() {
            log("DM_Facebook", 'Call "init"');
            var a = b.appId || DM_Widget_PageItem_Facebook_Connect_APP_ID;
            FB.init({
                appId: a,
                status: true,
                cookie: true,
                xfbml: true,
                oauth: true
            });
            log("DM_Facebook", "About to check FB user status");
            FB.getLoginStatus(b.handleSessionResponse);
            b.initDone = true
        },
        onFBScriptLoaded: function() {
            if (b.initDone) {
                return 
            }
            log("DM_Facebook", 'Call "onFBScriptLoaded"');
            b.init()
        },
        __onDomReady: function() {
            if (b.FBScriptLoaded) {
                log("DM_Facebook", "FB script is loaded");
                b.onFBScriptLoaded()
            } else {
                log("DM_Facebook", "FB script is NOT loaded yet")
            }
        },
        __initialize: function() {
            window.fbAsyncInit = function() {
                b.FBScriptLoaded = true;
                b.onFBScriptLoaded()
            };
            PubSub.subscribe("dm:social:mapped", b.onDailyMapped)
        }
    };
    return b
}, true);
_DM.define("DM_GooglePlus", function() {
    var c = ["https://www.googleapis.com/auth/plus.login", "https://www.googleapis.com/auth/userinfo.profile", "https://www.googleapis.com/auth/userinfo.email"], d = {
        connect: function() {
            log("DM_GooglePlus", "Try to connect to Google +");
            PubSub.publish("dm:gg:connecting");
            gapi.client.setApiKey(d._API_KEY);
            gapi.auth.authorize({
                client_id: d._CLIENT_ID,
                scope: c.join(" "),
                response_type: "code",
                immediate: false
            }, d.onLogin)
        },
        disconnect: function(b) {
            log("DM_GooglePlus", "Disconnect");
            if (b == "unknown") {
                PubSub.publish("dm:gg:status:unlogged", {
                    provider: "googleplus"
                })
            } else {
                if (b == "not_authorized") {
                    PubSub.publish("dm:gg:status:not_authorized")
                }
            }
            var a = DM_SocialAuth.getSocialCookie();
            if (a && typeof a.googleplus != "undefined") {
                delete a.googleplus
            }
            setCookie("dm_sc", $.toJSONString(a))
        },
        unlink: function(a) {
            PubSub.publish("dm:gg:unlinking");
            d.removePermissions(function() {
                ajax_call("social", "unlinkAccount", $.toJSONString({
                    provider: "googleplus",
                    userLogin: a
                }), function(b) {
                    d.disconnect();
                    PubSub.publish("dm:gg:unlinked", b)
                })
            })
        },
        checkAuth: function() {
            log("DM_GooglePlus", "Check authentication");
            gapi.auth.authorize({
                client_id: d._CLIENT_ID,
                scope: c.join(" "),
                response_type: "code",
                immediate: true
            }, d.onCheckAuth)
        },
        isAuthIdMatches: function(b) {
            var a = $.Deferred();
            gapi.client.load("oauth2", "v2", function() {
                var f = gapi.client.oauth2.userinfo.get();
                f.execute(function(e) {
                    log("DM_GooglePlus", "Get user GooglePlus info");
                    log("DM_GooglePlus", e);
                    if (parseInt(e.id, 10) == parseInt(b, 10)) {
                        a.resolve()
                    } else {
                        a.reject()
                    }
                })
            });
            return a.promise()
        },
        getMappedSocialData: function(b) {
            var a = {
                provider: "googleplus",
                code: b
            };
            ajax_call("social", "getMappedSocialData", $.toJSONString(a), d.afterMapping)
        },
        updateSocialCookie: function(a) {
            log("DM_GooglePlus", "User current Google+ session matches with DM user data");
            PubSub.publish("dm:gg:status:logged", {
                onLoginOrRegister: false,
                provider: "googleplus"
            });
            var b = DM_SocialAuth.getSocialCookie();
            if (typeof b.googleplus.authParams == "undefined") {
                b.googleplus = $.extend(b.googleplus, a);
                setCookie("dm_sc", $.toJSONString(b));
                PubSub.publish("dm:social:cookie:updated", "googleplus")
            }
        },
        onDailyMapped: function(b) {
            if (b.provider != "googleplus") {
                return 
            }
            var f = DM_SocialAuth.getSocialCookie(), a = f.googleplus || {};
            f.googleplus = $.extend(b.socialData, a);
            setCookie("dm_sc", $.toJSONString(f));
            PubSub.publish("dm:social:cookie:updated", "googleplus")
        },
        onCheckAuth: function(a) {
            if (a.error) {
                log("DM_GooglePlus", "Authentication failed, access denied");
                PubSub.publish("dm:gg:status:unlogged", {
                    provider: "googleplus"
                })
            } else {
                log("DM_GooglePlus", "Authentication succeed, set cookie data");
                var b = DM_SocialAuth.getSocialCookie();
                b.googleplus = {
                    session_state: a.session_state,
                    token_time: new Date().getTime(),
                    expires_in: a.expires_in
                };
                setCookie("dm_sc", $.toJSONString(b));
                d.getMappedSocialData(a.code)
            }
        },
        onSessionChecked: function(a) {
            if (!getCookie("sid")) {
                return 
            }
            log("DM_GooglePlus", a);
            if (!a || a.error) {
                log("DM_GooglePlus", "User is not logged on Google+ OR has not authorized the app");
                return d.disconnect("unknown")
            }
            log("DM_GooglePlus", "User is logged on GooglePlus");
            PubSub.publish("dm:gg:status:logged", {
                onLoginOrRegister: false,
                provider: "googleplus"
            })
        },
        onLogin: function(a) {
            if (!a.error) {
                log("DM_GooglePlus", 'User logged on Google+ and linked with "Daily Google+ Sign in"');
                PubSub.publish("dm:gg:status:logged", {
                    onLoginOrRegister: true,
                    provider: "googleplus"
                });
                var b = DM_SocialAuth.getSocialCookie();
                b.googleplus = {
                    session_state: a.session_state,
                    token_time: new Date().getTime(),
                    expires_in: a.expires_in
                };
                setCookie("dm_sc", $.toJSONString(b));
                return DM_SocialAuth.afterSocialLogged({
                    provider: "googleplus",
                    code: a.code
                })
            }
        },
        afterMapping: function(h) {
            log("DM_GooglePlus", "Need to validate access token");
            log("DM_GooglePlus", $.parseJSON(h));
            var a = $.parseJSON(h), b = $.parseJSON(a.authParams.token), g = b.access_token;
            $.ajax("https://www.googleapis.com/oauth2/v1/tokeninfo", {
                data: {
                    access_token: g
                },
                success: function(e) {
                    if (e.error) {
                        log("DM_GooglePlus", "Error when validating token");
                        return d.disconnect("unknown")
                    } else {
                        gapi.auth.setToken(b);
                        log("DM_GooglePlus", "Token validated, need to check if logged and mapped DM User is still connected to the right Google+ account");
                        $.when(d.isAuthIdMatches(a.id)).done(function() {
                            d.updateSocialCookie($.parseJSON(h))
                        }).fail(function() {
                            d.disconnect("unknown")
                        })
                    }
                }
            })
        },
        removePermissions: function(a) {
            log("DM_GooglePlus", "remove Dailymotion permissions on user app settings");
            var b = DM_SocialAuth.getSocialCookie(), h = $.parseJSON(b.googleplus.authParams.token).access_token, g = "https://accounts.google.com/o/oauth2/revoke?token=" + h;
            $.ajax({
                type: "GET",
                url: g,
                async: false,
                contentType: "application/json",
                dataType: "jsonp",
                success: function(e) {
                    a()
                },
                error: function(e) {
                    dbg(e)
                }
            })
        },
        onScriptLoaded: function() {
            if (d.initDone) {
                return 
            }
            if (!getCookie("sid")) {
                log("DM_GooglePlus", "User is not logged into DM");
                return 
            }
            if (!d.readerIsMapped) {
                log("DM_GooglePlus", "User is not mapped to Google");
                return 
            }
            var f = DM_SocialAuth.getSocialCookie();
            log("DM_GooglePlus", 'Call "onScriptLoaded"');
            if (typeof f.googleplus != "undefined") {
                var a = new Date().getTime();
                if (a - f.googleplus.token_time > (f.googleplus.expires_in)) {
                    log("DM_GooglePlus", "The token has expired, need a refresh token");
                    d.checkAuth()
                } else {
                    log("DM_GooglePlus", "Check user Google+ session state");
                    var b = f.googleplus.session_state;
                    gapi.auth.checkSessionState({
                        client_id: d._CLIENT_ID,
                        session_state: b
                    }, d.onSessionChecked)
                }
                d.initDone = true;
                return 
            }
            log("DM_GooglePlus", "Need to authorize the user to the app");
            gapi.client.setApiKey(d._API_KEY);
            d.checkAuth();
            d.initDone = true
        },
        __onDomReady: function() {
            if (d.scriptLoaded) {
                log("DM_GooglePlus", "Google+ script is loaded");
                d.onScriptLoaded()
            } else {
                log("DM_GooglePlus", "Google+ script is NOT loaded yet")
            }
        },
        __initialize: function() {
            window.ggAsynInit = function() {
                d.scriptLoaded = true;
                d.onScriptLoaded()
            };
            PubSub.subscribe("dm:social:mapped", d.onDailyMapped)
        }
    };
    return d
}, true);
_DM.define("Sd_Social_ConnectButton", function() {
    var c = {
        facebook: false,
        googleplus: false
    }, d = {
        onClicked: function() {
            var a = $(this).data("provider");
            if (!c[a]) {
                c[a] = true;
                var b = $(this).data("login-params") || "";
                if ($(this).data("no-reload")) {
                    b.noReload = $(this).data("no-reload")
                }
                DM_SocialAuth.connect(a, b)
            }
        },
        onConnecting: function(a) {
            c[a] = false
        },
        __onDomReady: function() {
            $("#bodyall, .dm_page_html").on("click", ".js-btn-googleplus, .js-btn-facebook", d.onClicked)
        },
        __initialize: function() {
            PubSub.subscribe("dm:fb:connecting", function() {
                d.onConnecting("facebook")
            });
            PubSub.subscribe("dm:gg:connecting", function() {
                d.onConnecting("googleplus")
            })
        }
    };
    return d
});
_DM.define("DM_Widget_PageItem_Notice", function() {
    var b = {
        initEvents: function() {
            $("div.dmpi_notice").each(function() {
                var a = $(this);
                if (a.hasClass("appear")&&!a.is(":visible")) {
                    a.fadeToggle(700)
                }
            }).on("click", "div.close_notice", DM_Widget_PageItem_Notice.onClose).on("click", "div.permanent_close_notice", DM_Widget_PageItem_Notice.onPermanentClose)
        },
        onClose: function() {
            PubSub.publish("notice:close", {
                name: $(this).data("name")
            });
            $(this).closest(".dmpi_notice").fadeToggle(400)
        },
        onPermanentClose: function(a) {
            $(this).closest(".dmpi_notice").fadeToggle(400);
            setCookie($(this).findParamInClass("name"), "true", 365)
        },
        __initialize: function() {
            $(function() {
                b.initEvents()
            })
        }
    };
    return b
}, true);
_DM.define("DM_Widget_PageItem_Search", function() {
    var b = {
        version: 5,
        autocomplete_enabled: "0",
        language: "en",
        historyCounter: 0,
        checkUserHistory: function(i) {
            try {
                var j = i.typedValue;
                if (localStorage && localStorage.hasOwnProperty("search_history") && j.length > 1) {
                    var e = JSON.parse(localStorage.getItem("search_history"));
                    var a = 0;
                    $(e).each(function(d, c) {
                        if (c.substring(0, j.length) == j) {
                            b.checkDuplicateValue(c);
                            i.suggestions.splice(a, 0, [c, 0, 0, [[j.length, c.length - j.length]], 0, false]);
                            a++
                        }
                        if (a == 2) {
                            b.historyCounter = a;
                            return false
                        }
                    });
                    b.historyCounter = a
                }
            } catch (h) {}
        },
        checkDuplicateValue: function(a) {
            $(b.autocomplete.suggestions).each(function(e, f) {
                if (f[0] == a) {
                    b.autocomplete.suggestions.splice(e, 1);
                    return false
                }
            })
        },
        updateInput: function() {
            this.options.appendMode = this.suggestions[this.selectedIndex] && (this.suggestions[this.selectedIndex][4] > 0);
            return Autocomplete.prototype.updateInput.apply(this, arguments)
        },
        suggest: function() {
            var j = b.autocomplete;
            if (j.suggestions.length === 0) {
                j.updateShadow();
                log("Search", "no suggestions to display, hide container");
                j.hide();
                return 
            }
            var k = [];
            var a = "";
            k.push('<div class="suggestions foreground">');
            b.checkUserHistory(j);
            $(j.suggestions).each(function(e, r) {
                a = "";
                boldString = "";
                var q = 0, c = r[3];
                for (var f = 0; f < c.length; f++) {
                    var d = c[f];
                    if (q < d[0]) {
                        a += r[0].slice(q, d[0])
                    }
                    q = d[0] + d[1];
                    boldString = r[0].slice(d[0], q);
                    a += "<b>" + boldString + "</b>"
                }
                if (q < r[0].length) {
                    a += r[0].slice(q)
                }
                var g = j.selectedIndex === e ? "selected": "";
                g = b.historyCounter > 0 ? "history" : g;
                if (b.historyCounter > 0) {
                    b.historyCounter--
                }
                k.push('<div class="js-suggestion_container ', g, '"', ' onclick="Autocomplete.instances[', j.instanceId, "].onSelect(", e, ')"', ' onmouseover="Autocomplete.instances[', j.instanceId, "].activate(", e, ')">', '<span class="suggestion" id="suggestion_', e, '"', ">", a, "</span>");
                k.push('<div class="clear_suggestion"></div></div>');
                $(j.inputRuler).html(r[4] > 0 ? j.inputValue.slice(0, r[4]).replace(/ /g, "&nbsp;") : "");
                if (e === 0) {
                    if (r[5] ||!boldString) {
                        j.updateShadow()
                    } else {
                        j.updateShadow(j.inputValue + boldString)
                    }
                } else {
                    if (e == 9) {
                        return false
                    }
                }
            });
            k.push("</div>");
            $(j.container).html(k.join("")).show();
            var l = $(j.el), h = $(j.el).position(), i = {
                left: h.left + "px",
                width: $(j.el).outerWidth() + "px"
            };
            if (j.inputRuler && $(j.inputRuler).width() !== 0) {
                i = {
                    left: (h.left + $(j.inputRuler).width()) + "px",
                    width: "auto"
                }
            }
            $(j.container).css(i);
            $(j.el).addClass("no_border_bottom_left_radius")
        },
        postConfigureSearchField: function(e) {
            var a = $("<div>").attr("id", "input_ruler"), f = $(e.el);
            e.inputRuler = a[0];
            f.after(a.css({
                position: "absolute",
                visibility: "hidden",
                padding: 0,
                margin: 0,
                font: f.css("font")
            }))
        },
        __initialize: function() {
            $(function() {
                if (window.DM_Widget_PageItem_Search_autocomplete_enabled == 1) {
                    var d = "/json/complete/" + DM_Widget_PageItem_Autocomplete_locale, a = $("#input_search");
                    if (a[0]) {
                        b.autocomplete = new Autocomplete("input_search", {
                            allowRefineLocally: false,
                            serviceUrl: d,
                            deferRequestBy: 10,
                            maxChars: 30,
                            postConfigureSearchField: b.postConfigureSearchField,
                            cleanStringOnValueChange: false,
                            cleanStringOnProcessResponse: false,
                            withShadow: true,
                            withIframeLayer: true,
                            suggest: b.suggest,
                            updateInput: b.updateInput
                        })
                    }
                }
            })
        }
    };
    return b
}, true);
_DM.define("Autocomplete", function() {
    var b = function(f, a) {
        if (!b.instances) {
            b.instances = []
        }
        this.instanceId = b.instances.push(this) - 1;
        this.searchFieldId = f;
        this.suggestionsId = "autocomplete_" + f;
        this.container = null;
        this.suggestions = [];
        this.$iframeLayer = null;
        this.selectedIndex =- 1;
        this.boundKiller = null;
        this.ignoreValueChange = false;
        this.cachedResponse = [];
        this.onChangeInterval = null;
        this.typedValue = "";
        this.inputValue = "";
        this.lastTypedValue = "";
        this.refineLocally = false;
        this.dataValue = null;
        this.shadowTimeout = null;
        this.options = {
            serviceUrl: "",
            autoSubmit: true,
            minChars: 1,
            maxChars: 0,
            deferRequestBy: 0,
            showInfo: true,
            autoPosition: false,
            appendMode: false,
            allowRefineLocally: true,
            suggestMax: 10,
            headerOffset: 36,
            suggestHeight: 20,
            postConfigureSearchField: null,
            withShadow: false,
            withIframeLayer: false,
            cleanStringOnValueChange: true,
            cleanStringOnProcessResponse: true,
            processResponse: null,
            suggest: null,
            onKeyUp: null,
            suggestionsProvider: null
        };
        for (var e in a) {
            if (!a.hasOwnProperty(e)) {
                continue
            }
            if (e in this) {
                this[e] = a[e];
                delete (a[e])
            }
        }
        $.extend(this.options, a);
        this.configureSearchField()
    };
    b.prototype = {
        version: 3,
        configureSearchField: function() {
            log("_self", "configureSearchField()");
            var a = $("#" + this.searchFieldId);
            this.el = a[0];
            a.attr("autocomplete", "off");
            a.on("keydown", $.proxy(this.onKeyDown, this)).on("keyup", $.proxy(this.onKeyUp, this)).on("blur", $.proxy(this.enablekiller, this));
            if (this.options.withShadow) {
                var d = a.position();
                $shadowEl = $('<input id="input_search_shadow" type="text" tabindex="-1" />').show().css({
                    left: d.left + "px",
                    top: d.top + "px",
                    width: a.width("width") + "px"
                });
                this.shadowEl = $shadowEl[0];
                a.addClass("has_shadow").after($shadowEl)
            }
            if (this.options.postConfigureSearchField && typeof this.options.postConfigureSearchField === "function") {
                this.options.postConfigureSearchField(this)
            }
        },
        createSuggestionsContainer: function() {
            log("_self", "createSuggestionsContainer()");
            var f = $("<div>"), a = {
                display: "none"
            }, e = $(this.el);
            if (this.options.withIframeLayer) {
                this.$iframeLayer = $('<iframe class="search_iframe_layer">')
            }
            f.addClass("autocomplete").attr("id", this.suggestionsId);
            if (this.options.autoPosition) {
                a.top = (e.position().top + e.height() + parseInt(e.css("padding-top"), 10) + parseInt(e.css("padding-bottom"), 10) + 2) + "px"
            }
            e.after(f.css(a));
            if (this.options.withIframeLayer) {
                e.before(this.$iframeLayer.css("top", f.css("top")))
            }
            this.container = $("#" + this.suggestionsId)[0]
        },
        onKeyDown: function(a) {
            if (this.options.withShadow) {
                if (this.shadowEl.value.indexOf(this.el.value + String.fromCharCode(a.keyCode).toLowerCase())===-1) {
                    this.updateShadow()
                }
            }
            switch (a.keyCode) {
            case 27:
                this.el.value = this.inputValue;
                this.hide();
                break;
            case 13:
                if (!this.options.autoSubmit) {
                    a.preventDefault()
                }
                this.onSelect(this.selectedIndex);
                break;
            case 38:
                this.moveUp();
                break;
            case 40:
                this.moveDown();
                break;
            case 8:
            case 32:
                this.updateShadow();
                return;
            default:
                return 
            }
            a.preventDefault()
        },
        onKeyUp: function(a) {
            switch (a.keyCode) {
            case 38:
            case 40:
            case 13:
                return;
            default:
                break
            }
            clearInterval(this.onChangeInterval);
            this.ignoreValueChange = false;
            if (this.typedValue !== this.el.value.toLowerCase()) {
                if (this.options.deferRequestBy > 0) {
                    this.onChangeInterval = setInterval($.proxy(this.onValueChange, this), this.options.deferRequestBy)
                } else {
                    this.onValueChange()
                }
            }
        },
        enablekiller: function(a) {
            if (this.boundKiller === null) {
                this.boundKiller = $.proxy(this.killer, this);
                $(document).on("click", this.boundKiller)
            }
        },
        disablekiller: function() {
            if (this.boundKiller !== null) {
                $(document).off("click", this.boundKiller);
                this.boundKiller = null
            }
        },
        killer: function(a) {
            if (!$(a.target).closest("#" + this.suggestionsId)[0]&&!$(a.target).closest("#" + this.searchFieldId)[0]) {
                this.hide();
                this.disablekiller()
            }
        },
        hide: function() {
            this.selectedIndex =- 1;
            if (this.container) {
                $(this.container).hide();
                if (this.options.withIframeLayer) {
                    $(this.$iframeLayer).hide()
                }
                $("#input_search").removeClass("no_border_bottom_left_radius")
            }
        },
        updateInput: function(g, j) {
            log("_self", "update input text with new value", g);
            if (this.options.appendMode&&!j) {
                var a = this.el.value.split(" "), h = a[a.length - 1], i = new RegExp(h + "$");
                this.el.value = this.el.value.replace(i, g)
            } else {
                this.el.value = g
            }
            this.updateShadow(this.el.value)
        },
        moveUp: function() {
            this.updateShadow();
            if (this.selectedIndex===-1) {
                this.selectedIndex = this.suggestions.length;
                this.activate(this.selectedIndex - 1);
                if (this.suggestions[this.selectedIndex].name) {
                    this.updateInput(this.suggestions[this.selectedIndex].name)
                } else {
                    this.updateInput(this.suggestions[this.selectedIndex][0])
                }
                return 
            }
            if (this.selectedIndex === 0) {
                $(this.container).find(".selected").removeClass("selected");
                this.selectedIndex =- 1;
                this.updateInput(this.inputValue, true);
                return 
            }
            this.activate(this.selectedIndex - 1);
            if (this.suggestions[this.selectedIndex].name) {
                this.updateInput(this.suggestions[this.selectedIndex].name)
            } else {
                this.updateInput(this.suggestions[this.selectedIndex][0])
            }
        },
        moveDown: function() {
            this.updateShadow();
            if (this.selectedIndex === (this.suggestions.length - 1)) {
                $(this.container).find(".selected").removeClass("selected");
                this.selectedIndex =- 1;
                this.updateInput(this.inputValue, true);
                return 
            }
            this.activate(this.selectedIndex + 1);
            if (this.suggestions[this.selectedIndex].name) {
                this.updateInput(this.suggestions[this.selectedIndex].name)
            } else {
                this.updateInput(this.suggestions[this.selectedIndex][0])
            }
        },
        onValueChange: function() {
            clearInterval(this.onChangeInterval);
            this.lastTypedValue = this.typedValue.toLowerCase();
            this.typedValue = this.el.value.toLowerCase();
            this.inputValue = this.el.value;
            this.selectedIndex =- 1;
            log("_self", "onValueChange(), new value", this.typedValue);
            if (!$("#" + this.suggestionsId)[0]) {
                this.createSuggestionsContainer()
            }
            if (this.ignoreValueChange) {
                this.ignoreValueChange = false;
                return 
            }
            if (this.options.cleanStringOnValueChange) {
                this.typedValue = this.cleansSearchString(this.typedValue)
            }
            if (this.typedValue === "" || this.typedValue.length < this.options.minChars || (this.options.maxChars && this.typedValue.length > this.options.maxChars)) {
                log("_self", "typed value empty || < minChars || > maxChars");
                this.hide();
                this.updateShadow()
            } else {
                this.getSuggestions()
            }
        },
        onSelect: function(f, e) {
            var a = this.suggestions[f] || [this.cleansSearchString(this.el.value)];
            if (this.options.onSelectSuggestion && typeof this.options.onSelectSuggestion === "function") {
                this.options.onSelectSuggestion(a, this.suggestions, e)
            } else {
                if (a[0]) {
                    this.updateShadow();
                    this.updateInput(a[0]);
                    this.ignoreValueChange = true
                }
            }
            this.hide();
            if (this.options.autoSubmit && $(this.el).closest("form")[0]&&!e) {
                $(this.el).closest("form").submit();
                this.addRequestsToLocalStorage(a[0])
            }
        },
        addRequestsToLocalStorage: function(a) {
            try {
                if (localStorage) {
                    var e;
                    if (localStorage.hasOwnProperty("search_history")) {
                        e = JSON.parse(localStorage.getItem("search_history"));
                        if ($.inArray(a, e)>-1) {
                            e.splice($.inArray(a, e), 1)
                        }
                        e.unshift(a);
                        if (e.length > 50) {
                            e.pop()
                        }
                    } else {
                        e = [];
                        e.push(a)
                    }
                    localStorage.setItem("search_history", JSON.stringify(e))
                }
            } catch (f) {}
        },
        getSuggestions: function() {
            log("_self", "getSuggestions() for value", this.typedValue);
            if ($.isArray(this.cachedResponse[encodeURIComponent(this.accentsTidy(this.typedValue))])) {
                this.truncateSuggestions(this.cachedResponse[encodeURIComponent(this.accentsTidy(this.typedValue))]);
                log("_self", "found cached response for value", this.typedValue, this.suggestions);
                this._suggest()
            } else {
                if (this.refineLocally && this.options.allowRefineLocally) {
                    if (!this.lastTypedValue.match(/^\s*$/) && this.el.value.toLowerCase().indexOf(this.lastTypedValue) === 0) {
                        this.suggestions = $(this.suggestions).grep(function(a) {
                            return this.accentsTidy(a[0]).indexOf(this.accentsTidy(this.typedValue)) === 0
                        }, this);
                        this.cachedResponse[encodeURIComponent(this.accentsTidy(this.typedValue))] = this.suggestions;
                        this._suggest();
                        return 
                    } else {
                        this.refineLocally = false
                    }
                }
                this.suggestionsProvider()
            }
        },
        suggestionsProvider: function() {
            var d = getCookie("dmt"), a = this.options.serviceUrl + "/" + encodeURIComponent(this.typedValue) + (d ? "?" + d : "");
            $.getJSON(a).always($.proxy(this.processResponse, this, a))
        },
        _suggest: function() {
            this.suggest();
            if (this.options.withIframeLayer) {
                this.$iframeLayer.show().css({
                    left: $(this.container).css("left"),
                    width: $(this.container).outerWidth() + "px",
                    height: $(this.container).outerHeight() + "px"
                })
            }
        },
        suggest: function() {
            log("_self", "suggest() results for value", this.typedValue);
            if (this.suggestions.length === 0) {
                this.updateShadow();
                log("_self", "no suggestions to display, hide container");
                this.hide();
                return 
            }
            var g = [];
            var i = "";
            g.push('<div class="suggestions background foreground">');
            var h = this;
            $(this.suggestions).each(function(d, c) {
                if (d === 0) {
                    h.updateShadow(c[0])
                }
                i = c[0];
                g.push((h.selectedIndex === d ? '<div class="selected"' : "<div"), ' onclick="Autocomplete.instances[', h.instanceId, "].onSelect(", d, ')"', ' onmouseover="Autocomplete.instances[', h.instanceId, "].activate(", d, ')">', '<span class="suggestion" id="suggestion_', d, '"', ">", i, "</span>");
                g.push('<div class="clear_suggestion"></div></div>')
            });
            g.push("</div>");
            $(this.container).html(g.join("")).show();
            if (this.searchFieldId == "input_search") {
                $("#input_search").addClass("no_border_bottom_left_radius")
            }
            var j = $("#" + this.searchFieldId), a = j.position();
            $(this.container).css({
                left: a.left + "px",
                width: j.width() + "px"
            })
        },
        processResponse: function(l, m, n, a) {
            var i = l.substring(l.lastIndexOf("/") + 1, l.length);
            if (i.indexOf("?")!=-1) {
                i = i.substring(0, i.indexOf("?"))
            }
            log("_self", "processResponse()", m);
            this.cachedResponse[encodeURIComponent(this.accentsTidy(this.typedValue))] = m;
            this.truncateSuggestions(m);
            if (this.suggestions.length < 10 && this.options.allowRefineLocally) {
                this.refineLocally = true
            }
            var j = this.el.value.toLowerCase(), k = this.options.cleanStringOnProcessResponse ? this.cleansSearchString(j): j;
            if (i === encodeURIComponent(k)) {
                this._suggest()
            }
        },
        truncateSuggestions: function(j) {
            this.suggestions = j;
            var a = $("html");
            if (a.hasClass("win") && (a.hasClass("webkit") || a.hasClass("ie"))) {
                var g = $("#main_player") || $(".dmpi_video_playerv4"), i = g[0];
                if (i) {
                    var h = g.offset().top - $("#header").offset().top - this.options.headerOffset;
                    this.options.suggestMax = Math.floor(h / this.options.suggestHeight);
                    if (!$(document.body).hasClass("dm_page_html_video")) {
                        this.options.suggestMax--
                    }
                }
                if (this.options.suggestMax < 10) {
                    this.suggestions = j.slice(this.options.suggestMax - 1)
                }
            }
        },
        activate: function(a) {
            var e;
            if (this.container && $(this.container).is(":visible")) {
                $(this.container).find(".selected").removeClass("selected");
                this.selectedIndex = a;
                var f = $(this.container).children().children();
                if (this.selectedIndex!==-1 && f.length > this.selectedIndex) {
                    e = f[this.selectedIndex];
                    $(e).addClass("selected")
                }
            }
            return e
        },
        updateShadow: function(a) {
            if (this.options.withShadow) {
                var d = this;
                clearTimeout(this.shadowTimeout);
                if (a) {
                    this.shadowTimeout = setTimeout(function() {
                        d.shadowEl.value = a;
                        log("_self", "update shadow with value", a)
                    }, 300)
                } else {
                    this.shadowEl.value = "";
                    log("_self", "update shadow with value", a)
                }
            }
        },
        cleansSearchString: function(a) {
            return a.replace(/\s{2,}/g, " ").replace(/^\s+/g, "").replace(/\//g, "")
        },
        accentsTidy: function(a) {
            return a.toLowerCase().replace(/[àáâãäå]/g, "a").replace(/ç/g, "c").replace(/[èéêë]/g, "e").replace(/[ìíîï]/g, "i").replace(/ñ/g, "n").replace(/[òóôõö]/g, "o").replace(/[ùúûü]/g, "u").replace(/[ýÿ]/g, "y")
        }
    };
    return b
}, true);
_DM.define("DM_Tealium", function() {
    var b = {
        __initialize: function() {
            PubSub.subscribe("stats:event", b.onEvent)
        },
        onEvent: function(a) {
            if ("utag" in window) {
                switch (a.event_action) {
                case"share":
                case"follow":
                case"unfollow":
                case"upload":
                case"register":
                case"login":
                case"playlist":
                case"customer_survey":
                case"pixelle":
                case"expand_player":
                case"use_addto_tab":
                case"add_to_favourite":
                case"add_to_contest":
                case"add_to_group":
                case"add_to_watchlater":
                case"report_video":
                case"comment_video":
                case"promote_its_account":
                case"navigation_panel":
                    utag.link(a);
                    break
                }
            }
        }
    };
    return b
}, true);
_DM.define("DM_Krux", function() {
    var b = {
        __initialize: function() {
            if (DM_Context.geoip_country != "fr") {
                PubSub.subscribe("stats:event", b.onEvent)
            }
        },
        allowedEvents: {
            expand_player: "I-fDdTS1",
            share: "JCq10IJg",
            follow: "JCrBcaU_",
            comment_video: "JCrBmxti",
            report_video: "JCrB1GJw",
            add_to_watchlater: "JCrCESJe",
            playlist: "JCrB935k"
        },
        onEvent: function(j) {
            var h = j.event_action, a = j.event_context, i = j.event_data;
            if (b.allowedEvents[h] && typeof Krux !== "undefined") {
                switch (h) {
                case"share":
                    var g = {
                        twitter: "JCq17ViP",
                        blogger: "JCq2BEOW",
                        wordpress: "JCq2IJwY",
                        tumblr: "JCq2OHU6",
                        reddit: "JCq2UL-i",
                        pinterest: "JCrBId4p",
                        facebook: "JCrBP5pP",
                        use_embed_code: "JCrBWSM_"
                    };
                    if (g[a]) {
                        Krux("admEvent", g[a], {})
                    }
                    Krux("admEvent", b.allowedEvents[h], {});
                    break;
                case"follow":
                    switch (a) {
                    case null:
                    case"from_channel_tooltip":
                    case"with_push_tooltip":
                    case"unlogged":
                        Krux("admEvent", b.allowedEvents[h], {});
                        break
                    }
                    break;
                case"playlist":
                    if (a === "add_video") {
                        Krux("admEvent", b.allowedEvents[h], {})
                    }
                    break;
                default:
                    Krux("admEvent", b.allowedEvents[h], {});
                    break
                }
            }
        }
    };
    return b
}, true);
_DM.define("DM_SiteCatalyst", function() {
    var b = {
        onEvent: function(z) {
            if ("SiteCatalyst" in window) {
                var a = window.SiteCatalyst.getInstance(), w = z.event_action, y = z.event_context, o = z.event_label, s = z.event_data, p = a.prop18, u = a.prop19, t = a.prop22;
                a.events = false;
                switch (w) {
                case"share":
                    a.eVar12 = o;
                    a.eVar25 = t;
                    a.linkTrackVars = "eVar12,eVar25,events";
                    a.linkTrackEvents = a.events = "event7";
                    a.tl(true, "o", o);
                    break;
                case"follow":
                    switch (y) {
                    case null:
                    case"from_channel_tooltip":
                    case"with_push_tooltip":
                    case"unlogged":
                        a.eVar40 = o;
                        a.linkTrackVars = "eVar40,events";
                        a.linkTrackEvents = a.events = y == "unlogged" ? "event91" : "event2";
                        break;
                    case"push_tooltip_displayed":
                    case"push_tooltip_closed":
                        a.linkTrackVars = "events";
                        a.linkTrackEvents = a.events = y == "push_tooltip_displayed" ? "event86" : "event87";
                        break
                    }
                    if (a.events) {
                        a.tl(true, "o", o)
                    }
                    break;
                case"unfollow":
                    a.linkTrackVars = "events";
                    a.linkTrackEvents = a.events = "event3";
                    a.tl(true, "o", o);
                    break;
                case"upload":
                    switch (y) {
                    case"request":
                        a.events = "event32";
                        break;
                    case"start":
                        a.events = "event49";
                        break;
                    case"success":
                        a.events = "event25";
                        break;
                    case"fail":
                        a.events = "event54";
                        break;
                    case"save_edit_form":
                        a.events = "event51";
                        break
                    }
                    if (a.events) {
                        a.linkTrackVars = "events";
                        a.linkTrackEvents = a.events;
                        a.tl(true, "o", o)
                    }
                    break;
                case"register":
                    var r = y.split("-")[0], x = y.split("-")[1];
                    if (r == "dailymotion") {
                        switch (x) {
                        case"form_step1":
                            a.events = "event62";
                            break;
                        case"form_step2":
                            a.events = "event63,event26";
                            break;
                        case"take_a_tour":
                            a.events = "event64";
                            break;
                        case"areas_of_interest":
                            a.events = "event65";
                            break;
                        case"skip_facebook_connect":
                            a.events = "event66";
                            break;
                        case"follow_your_friends":
                            a.events = "event67";
                            break;
                        case"complete_tour":
                            a.events = "event68";
                            break;
                        case"close_tour":
                            a.events = "event69";
                            break;
                        default:
                        }
                    } else {
                        switch (x) {
                        case"facebook_connect":
                            a.events = "event70";
                            break;
                        case"form_step2":
                            a.events = "event71,event26";
                            break;
                        case"take_a_tour":
                            a.events = "event72";
                            break;
                        case"areas_of_interest":
                            a.events = "event73";
                            break;
                        case"follow_your_friends":
                            a.events = "event74";
                            break;
                        case"complete_tour":
                            a.events = "event75";
                            break;
                        case"close_tour":
                            a.events = "event76";
                            break;
                        default:
                        }
                    }
                    if (a.events) {
                        a.eVar41 = r.charAt(0).toUpperCase() + r.slice(1);
                        a.linkTrackVars = "eVar41,events";
                        a.linkTrackEvents = a.events;
                        a.tl(true, "o", o)
                    }
                    break;
                case"login":
                    a.eVar39 = o;
                    a.linkTrackVars = "eVar39,events";
                    a.linkTrackEvents = a.events = "event27";
                    a.tl(true, "o", o);
                    break;
                case"playlist":
                    switch (y) {
                    case"create":
                        a.events = "event28";
                        break;
                    case"add_video":
                        a.events = "event29";
                        break
                    }
                    if (a.events) {
                        a.eVar25 = t;
                        a.linkTrackVars = "eVar25,events";
                        a.linkTrackEvents = a.events;
                        a.tl(true, "o", o)
                    }
                    break;
                case"customer_survey":
                    switch (y) {
                    case"display_survey":
                        a.events = "event55";
                        break;
                    case"skip_survey":
                        a.events = "event56";
                        break
                    }
                    if (a.events) {
                        a.linkTrackVars = "events";
                        a.linkTrackEvents = a.events;
                        a.tl(true, "o", o)
                    }
                    break;
                case"pixelle":
                    switch (y) {
                    case"display":
                        a.events = "event96";
                        break;
                    case"click":
                        a.events = "event97";
                        break
                    }
                    a.linkTrackVars = "events";
                    a.linkTrackEvents = a.events;
                    a.tl(this, "o", o);
                    break;
                case"navigation_panel":
                    switch (y) {
                    case"generated_traffic":
                        a.events = "event92";
                        a.eVar74 = "content:" + s.content + ",link:" + s.link;
                        a.linkTrackVars = "eVar74,events";
                        break;
                    case"open":
                    case"close":
                        a.events = y === "open" ? "event93" : "event94";
                        a.linkTrackVars = "events";
                        break
                    }
                    if (a.events) {
                        a.linkTrackEvents = a.events;
                        a.tl(this, "o", o)
                    }
                    break;
                case"promote_its_account":
                    a.eVar21 = p;
                    a.eVar22 = u;
                    a.linkTrackVars = "eVar21,eVar22,events";
                    a.linkTrackEvents = a.events = "event40";
                    a.tl(true, "o", o);
                    break;
                case"report_video":
                case"comment_video":
                case"share_new_video_comment_on_facebook":
                case"expand_player":
                case"add_to_favourite":
                case"add_to_watchlater":
                case"add_to_group":
                case"add_to_contest":
                case"use_addto_tab":
                    switch (w) {
                    case"report_video":
                        a.events = "event43";
                        break;
                    case"comment_video":
                        a.events = "event44";
                        break;
                    case"expand_player":
                        a.events = "event48";
                        break;
                    case"add_to_favourite":
                        a.events = "event34";
                        break;
                    case"add_to_watchlater":
                        a.events = "event35";
                        break;
                    case"add_to_group":
                        a.events = "event36";
                        break;
                    case"add_to_contest":
                        a.events = "event37";
                        break;
                    case"use_addto_tab":
                        a.events = "event77";
                        break;
                    case"share_new_video_comment_on_facebook":
                        a.events = "event95";
                        break
                    }
                    if (a.events) {
                        a.eVar25 = t;
                        a.linkTrackVars = "eVar25,events";
                        a.linkTrackEvents = a.events;
                        a.tl(true, "o", o)
                    }
                    break;
                case"ligue1":
                    var q = s.content ? s.content: a.pageName, v = "Lightbox Video Ligue 1";
                    ajax_call("sitecatalyst", "get_ajax_page_code", q, v, function(c) {
                        if (c) {
                            $("#sitecatalyst").html(c)
                        }
                    });
                    break;
                case"oauth":
                    switch (y) {
                    case"login":
                        a.events = "event1";
                        break;
                    case"signup":
                        a.events = "event2";
                        break;
                    case"password":
                        a.events = "event3";
                        break;
                    case"facebook_login":
                        a.events = "event4";
                        break;
                    case"facebook_signup":
                        a.events = "event5";
                        break
                    }
                    if (DM_Context.traffic_segment) {
                        a.eVar48 = DM_Context.traffic_segment
                    } else {
                        a.eVar48 =- 1
                    }
                    if (a.events) {
                        a.events = a.events + ",event6";
                        a.linkTrackVars = "eVar48,events";
                        a.linkTrackEvents = a.events;
                        a.tl(this, "o", o)
                    }
                    break
                }
            }
        },
        __initialize: function() {
            $(function() {
                PubSub.subscribe("stats:event", b.onEvent)
            })
        }
    };
    return b
}, true);
_DM.define("DM_Stats", function() {
    var b = {
        formattedEvent: false,
        events: {
            share: {
                share: "Share",
                twitter: "Twitter (Official)",
                blogger: "Blogger (AddThis)",
                wordpress: "Wordpress (AddThis)",
                tumblr: "Tumblr (AddThis)",
                reddit: "Reddit (AddThis)",
                pinterest: "Pinterest (AddThis)",
                use_embed_code: "Use embed code video panel"
            },
            follow: {
                follow: "Follow",
                from_channel_tooltip: "Follow from a channel tooltip",
                with_push_tooltip: "Follow with push tooltip displayed",
                unlogged: "Follow when user is unlogged",
                display_push_tooltip: "Display push follow tooltip",
                close_push_tooltip: "Close push follow tooltip"
            },
            unfollow: "Unfollow",
            upload: {
                request: "Upload request",
                start: "Upload start",
                success: "Upload success",
                fail: "Upload fail",
                save_edit_form: "Save Edit form"
            },
            register: {
                "dailymotion-form_step1": "DM - Submit register form - Step 1",
                "dailymotion-form_step2": "DM - Submit register form - Step 2 - Create account",
                "dailymotion-take_a_tour": "DM - Take the welcome tour",
                "dailymotion-areas_of_interest": "DM - Tour (areas of interest) next",
                "dailymotion-skip_facebook_connect": "DM - Tour - Skip Facebook Connect",
                "dailymotion-facebook_follow_your_friends": "DM - Tour (Facebook Friends) next",
                "dailymotion-complete_tour": "DM - Complete the tour",
                "dailymotion-close_tour": "DM - Close the tour",
                "facebook-connect": "FB - Facebook Connect submitted",
                "facebook-form_step2": "FB - Submit register form - Step 2 - Create account",
                "facebook-take_a_tour": "FB - Take the welcome tour",
                "facebook-areas_of_interest": "FB - Tour (areas of interest) next",
                "facebook-follow_your_friends": "FB - Tour (Facebook Friends) next",
                "facebook-complete_tour": "FB - Complete the tour",
                "facebook-close_tour": "FB - Close the tour"
            },
            login: {
                dailymotion: "Login via Dailymotion",
                facebook: "Login via Facebook"
            },
            playlist: {
                add_video: "Add video to playlist",
                create: "Create playlist"
            },
            customer_survey: {
                display: "Display customer survey",
                skip: "Skip customer survey"
            },
            pixelle: {
                display: "Display a promoted video",
                click: "Click on a promoted video"
            },
            promote_its_account: "Promote its account",
            report_video: "Report video",
            comment_video: "Comment video",
            share_new_video_comment_on_facebook: "Share new video comment on Facebook",
            expand_player: "Expand player",
            add_to_favourite: "Add video to favourite",
            add_to_watchlater: "Add video to Watch Later",
            add_to_group: "Add video to group",
            add_to_contest: "Add video to contest",
            use_addto_tab: "Click on AddTo tab in player page",
            navigation_panel: {
                open: "Open navigation menu",
                close: "Close navigation menu",
                generated_traffic: "Click on item from the navigation menu"
            },
            oauth: {
                sign_in: "Click on Login",
                sign_up: "Click on Signup",
                recover_password: "Click on password recover",
                facebook_sign_in: "Click on FB Login",
                facebook_sign_up: "Click on FB Signup"
            },
            ligue1: "Video view on Ligue1 page"
        },
        trigger: function(a) {
            b.getLabel(a);
            if (b.formattedEvent) {
                PubSub.publish("stats:event", b.formattedEvent)
            }
        },
        getLabel: function(d) {
            var a = false;
            if (b.events[d.action]) {
                if (!b.events[d.context] && typeof b.events[d.action] !== "object") {
                    a = b.events[d.action]
                } else {
                    if (!b.events[d.action][d.context]) {
                        a = b.events[d.action][d.action]
                    } else {
                        a = b.events[d.action][d.context]
                    }
                }
            }
            if (a) {
                b.formattedEvent = {
                    event_action: d.action,
                    event_context: d.context ? d.context: null,
                    event_label: a,
                    event_data: d.data ? d.data: {}
                }
            }
            return a
        }
    };
    return b
}, true);
var vsUrl, cookieEnabled = (function() {
    setCookieWithFormat("cke", "1", 1, "m");
    if (getCookie("cke") == "1") {
        deleteCookie("cke");
        return true
    }
    return false
})();
window.DM_PlayerEvents && DM_PlayerEvents.on("Ready", VS_onDailymotionPlayerReady);
function VS_onDailymotionPlayerReady(b) {
    DM_PlayerEvents.on("VideoMetadata", b, onVSPlayerVideoMetadata)
}
function onVSPlayerVideoMetadata() {
    if (window.logPageViewFromPlayer) {
        vsStat()
    }
}
function vsStat(b) {
    if (!cookieEnabled) {
        return 
    }
    if (vsUrl) {
        ping(vsUrl, "vs")
    }
}
$(function() {
    vsStat()
});
_DM.define("DM_LoggerEvent_Trigger", function() {
    var b = {
        onLoggerEventAction: function(h) {
            var g;
            if (typeof loggerEventServerDatas !== undefined && loggerEventServerDatas !== null) {
                g = $.parseJSON(loggerEventServerDatas)
            }
            if ($(this).data("logger-event") !== undefined) {
                h = $(this).data("logger-event")
            }
            var f = $.extend(g, h);
            if (f) {
                var a = new DM_LoggerEvent();
                a.setParams(f);
                a.ping()
            }
        },
        initEvents: function() {
            $("[data-logger-event]").on("click", DM_LoggerEvent_Trigger.onLoggerEventAction)
        },
        __initialize: function() {
            $(function() {
                PubSub.subscribe("loggerEvent", DM_LoggerEvent_Trigger.onLoggerEventAction);
                b.initEvents()
            })
        }
    };
    return b
}, true);
_DM.define("DM_LoggerEvent", function() {
    var b = (function() {
        var a = function() {
            this.url = null;
            this.mandatoryParams = ["host", "namespace", "clientId", "sourceId", "initiator", "action"];
            this.params = {};
            this.params.host = null;
            this.params.namespace = "dm";
            this.params.clientId = undefined;
            this.params.sourceId = undefined;
            this.params.initiator = "u";
            this.params.action = undefined
        };
        a.prototype.setParam = function(f, e) {
            this.params[f] = e
        };
        a.prototype.setParams = function(e) {
            for (var f in e) {
                this.setParam(f, e[f])
            }
        };
        a.prototype.computeURL = function() {
            var g, h = this;
            this.url = "http://" + this.params.host + "/" + this.params.namespace + "/" + this.params.clientId + "/" + this.params.sourceId + "/" + this.params.initiator + "/" + this.params.action;
            if ($.keys(this.params).length > this.mandatoryParams.length) {
                this.url += "?";
                for (var f in this.params) {
                    if (f == "host") {
                        g = this.params[f]
                    }
                    if (this.mandatoryParams.indexOf(f)>-1) {
                        delete this.params[f]
                    }
                }
                this.url += $.keys(this.params).map(function(c) {
                    return encodeURIComponent(c) + "=" + encodeURIComponent(h.params[c])
                }).join("&")
            }
            if (g) {
                return true
            } else {
                return false
            }
        };
        a.prototype.ping = function() {
            if (this.computeURL()) {
                $.ajax({
                    type: "GET",
                    url: this.url,
                    noFromRequest: true
                }).done(function(d) {
                    log("loggerEvent", this.url + " : Ping OK")
                }).fail(function(f, e) {
                    log("loggerEvent", this.url + " : Ping fail : " + e)
                })
            } else {
                log("loggerEvent", this.url + " : Ping fail : no logger host")
            }
        };
        return a
    })();
    return b
}, true);
_DM.define("DM_Ezakus", function() {
    var b = {
        __initialize: function() {
            if (DM_Context.geoip_country == "fr") {
                PubSub.subscribe("stats:event", b.onEvent)
            }
        },
        allowedEvents: ["expand_player", "share", "follow", "comment_video", "report_video", "add_to_watchlater", "playlist"],
        onEvent: function(j) {
            var h = j.event_action, a = j.event_context, i = j.event_data;
            if ($.inArray(h, b.allowedEvents)>-1 && typeof ezMarker !== "undefined") {
                switch (h) {
                case"share":
                    var g = ["twitter", "blogger", "wordpress", "tumblr", "reddit", "pinterest", "facebook", "publishit"];
                    if ($.inArray(a, g)>-1) {
                        ezMarker.event(h + "_" + a)
                    }
                    ezMarker.event(h);
                    break;
                case"follow":
                    switch (a) {
                    case null:
                    case"from_channel_tooltip":
                    case"with_push_tooltip":
                    case"unlogged":
                        ezMarker.event(h);
                        break
                    }
                    break;
                case"expand_player":
                    ezMarker.event("player_expand");
                    break;
                case"report_video":
                case"comment_video":
                    ezMarker.event(h.split("_")[0]);
                    break;
                case"playlist":
                    if (a === "add_video") {
                        ezMarker.event("add_to_playlist")
                    }
                    break;
                default:
                    ezMarker.event(h);
                    break
                }
            }
        }
    };
    return b
}, true);
_DM.define("Sd_Navigation_Panel", function() {
    var j = {
        contentOffset: false,
        bgPercentage: false,
        backgroundInfos: false,
        currentBgPos: 50,
        bgPositionX: false
    }, g = 260, i = 970 + g;
    function k(b, e, d) {
        var a = d ? "prepend": "append", c = l(b, e.hasClass("activity"));
        e.find(".list-placeholder").remove();
        e[a](c).find("li").show();
        e.closest("section").show();
        PubSub.publish("dm:nav:listdisplayed", e)
    }
    function l(b, a) {
        var c = [], e = typeof b[0] !== "undefined" && typeof b[0].uri !== "undefined"&&!a, d = a ? 20: 25;
        b.forEach(function(p) {
            if (p.activity) {
                d = p.activity.toString().length > 1 ? d - p.activity.toString().length : d
            }
            var q = p.screenname.length > d ? p.screenname.substr(0, d - 3) + "...": p.screenname, f = "link-on-hvr" + (e ? " js-channel-tip" : ""), r = e ? 'data-user-uri="' + p.uri + '" data-channel-tip-gravity="w"': "";
            c.push('<li style="display: none"><a class="' + f + '" href="' + p.url + '" data-link="' + p.screenname + '" ' + r + ">");
            typeof p.avatar_60_url !== "undefined" && c.push('<div class="avatar-wrapper"><img class="brd-rad-sm" src="' + p.avatar_60_url + '"></div>');
            c.push(q);
            typeof p.activity !== "undefined" && c.push('<span class="pull-end ' + (p.activity > 0 ? "" : "hidden") + '">' + p.activity + "</span>");
            c.push("</a></li>")
        });
        return c.join("")
    }
    var h = {
        responsive: false,
        initDone: false,
        subscriptionsLoaded: false,
        subscriptionsPage: 1,
        blockStoreKey: "nav-block-status",
        blockStatus: {},
        resizeInterval: false,
        initNavigationPanel: function() {
            h.loadDatas();
            if (h.responsive_page) {
                i = 970 + 300 + ($("body").hasClass("skyscraper") ? 160 : 0)
            }
            if (i >= $(window).width() && h.responsive_page) {
                h.responsive = false
            }
            if (h.responsive) {
                var c = $("#navigation-panel");
                if (!$("#responsive-nav-panel").length) {
                    $('<div class="navigation-panel responsive" id="responsive-nav-panel"></div>').insertBefore($(".navigation-border"))
                }
                $("#responsive-nav-panel").prepend(c.html());
                c.remove();
                h.navigationPanel = $("#responsive-nav-panel")
            } else {
                h.navigationPanel = $("#navigation-panel")
            }
            j.bgPositionX = typeof $("body").css("background-position-x") !== "undefined" ? true : false;
            h.headerNavigationTrigger = $("#header-navigation-trigger");
            h.nav = h.navigationPanel.find("nav");
            h.navigationWrapper = $(".navigation-wrapper");
            h.subscriptionLoadMore = $("#subscription-load-more");
            h.navigationPanelVisible = false;
            if (h.responsive === false) {
                h.repositionNavigationPanel();
                h.resizeNavigationPanelHeight()
            }
            h.windowWidth = $(window).width();
            if (store.get("navigation:status") === "open" && h.windowWidth > i && h.isAutoOpenRoute()) {
                h.openNavigationPanel(0)
            } else {
                if (h.responsive && h.windowWidth > i) {
                    h.setCloseResponsiveAttributes()
                }
                $("body").removeClass("no-transition")
            }
            if (h.initDone) {
                return 
            }
            h.headerNavigationTrigger.on("click", h.toggleNavigationPanel);
            $(".navigation-sub-trigger").on("click", h.toggleSubMenu);
            h.subscriptionLoadMore.on("click", function(e) {
                h.store.set("nav-show-subscriptions", true);
                h.loadSubscriptions(e, true)
            });
            h.nav.find("a").on("click", function() {
                var e = $(this), f = e.closest("section").data("content"), n = e.data("link") ? e.data("link"): f;
                DM_Stats.trigger({
                    action: "navigation_panel",
                    context: "generated_traffic",
                    data: {
                        content: f,
                        link: n
                    }
                })
            });
            $(window).on({
                resize: function() {
                    if (h.responsive) {
                        DM_Utils.throttle(function() {
                            if (h.responsive === false) {
                                return 
                            }
                            if (i >= $(window).width() && h.responsive_page) {
                                $(".navigation-panel").detach().removeClass("responsive").addClass("fixed").removeAttr("id").attr("id", "navigation-panel").removeAttr("style").insertAfter(h.headerNavigationTrigger);
                                var e = h.navigationPanelVisible;
                                h.closeNavigationPanel();
                                h.responsive = false;
                                h.initNavigationPanel();
                                if (e) {
                                    h.openNavigationPanel()
                                }
                                PubSub.publish("player:resized")
                            }
                            h.resizeResponsiveMenu()
                        }, 250, true)
                    } else {
                        DM_Utils.throttle(function() {
                            if (i <= $(window).width() && h.responsive_page) {
                                var e = h.navigationPanelVisible;
                                h.closeNavigationPanel();
                                h.responsive = true;
                                h.initNavigationPanel();
                                if (e) {
                                    h.openNavigationPanel()
                                }
                            } else {
                                if (h.navigationPanelVisible) {
                                    h.windowWidth !== $(window).width() && h.repositionNavigationPanel();
                                    h.resizeNavigationPanelHeight()
                                }
                            }
                            h.windowWidth = $(window).width();
                            j.contentOffset = false
                        }, 250, true)
                    }
                },
                scroll: function() {
                    if (h.responsive) {
                        var e = $(window), n = $(".sd_header"), f = n.offset().top + n.height();
                        if (e.scrollTop() <= 0) {
                            h.navigationPanel.css("top", f + "px")
                        } else {
                            if (e.scrollTop() < f) {
                                h.navigationPanel.css("top", f - e.scrollTop() + "px")
                            } else {
                                h.navigationPanel.css("top", 0)
                            }
                        }
                    } else {
                        if (h.navigationPanelVisible) {
                            DM_Utils.throttle(h.resizeNavigationPanelHeight, 250, true);
                            h.windowWidth < i && h.closeNavigationPanel()
                        }
                    }
                }
            });
            PubSub.subscribe("watchlater:added", function() {
                h.updateWatchLater(true)
            }).subscribe("watchlater:removed", function() {
                h.updateWatchLater(false)
            }).subscribe("dm:tabs_initialized", function() {
                h.resizeResponsiveMenu()
            }).subscribe("dm:smartalert:displaying", h.forceResize).subscribe("dm:smartalert:displayed", h.endForceResize).subscribe("dm:smartalert:closing", h.forceResize).subscribe("dm:smartalert:closed", h.endForceResize).subscribe("dm:metaheader:closed", function() {
                h.navigationPanelVisible && h.resizeNavigationPanelHeight()
            }).subscribe("dm:navigation:content:added", h.repositionContent);
            var b = store.get(h.blockStoreKey);
            h.blockStatus = typeof b === "object" ? b : {};
            for (var a in h.blockStatus) {
                var d = $('section[data-content="' + a + '"]').find(".navigation-sub-trigger");
                if (h.blockStatus[a]) {
                    d.hasClass("selected") || h.toggleSubMenu.call(d, false, false)
                } else {
                    h.toggleSubMenu.call(d, false, false)
                }
            }
            h.highlightCurrentLink();
            h.store.get("nav-hide-notif") && $("#header-navigation-trigger .navigation-notification").remove();
            h.initDone = true
        },
        forceResize: function() {
            h.resizeInterval = setInterval(function() {
                if (h.navigationPanelVisible) {
                    h.resizeNavigationPanelHeight(true)
                }
            }, 5)
        },
        endForceResize: function() {
            clearInterval(h.resizeInterval);
            h.resizeNavigationPanelHeight()
        },
        repositionNavigationPanel: function() {
            var a = 0, c = 0;
            var b = h.headerNavigationTrigger.offset();
            if (DM_Dir.start === "right") {
                if (h.windowWidth > (g + b.left)) {
                    c = h.windowWidth - (b.left + 130)
                }
                h.navigationPanel.css("right", c)
            } else {
                if (b.left > 130) {
                    a = b.left - 130
                }
                h.navigationPanel.css("left", a)
            }
            h.repositionContent(0)
        },
        getBackgroundSize: function() {
            if (j.backgroundInfos) {
                return {
                    then: function(d) {
                        d(j.backgroundInfos)
                    }
                }
            }
            var a = $("#topwrapper"), b = a.css("background-image").replace("url(", "").replace(")", "").replace('"', "");
            if (b === "none") {
                return {
                    then: function(d) {
                        d(false)
                    }
                }
            }
            j.contentOffset = false;
            var c = new Image();
            c.src = b;
            return {
                then: function(d) {
                    c.onload = function() {
                        j.backgroundInfos = {
                            width: c.width,
                            positionY: a.css("background-position").split(" ")[1]
                        };
                        d(j.backgroundInfos)
                    }
                }
            }
        },
        repositionContent: function(b, c) {
            b = typeof b !== "undefined" ? b : 100;
            c = c || false;
            var d = $("#content, #top_banner, .topwrapper_content, #content-bottom-content .container"), a = $("#topwrapper");
            h.getBackgroundSize().then(function(e) {
                if ((h.navigationPanelVisible || c) && h.windowWidth > i) {
                    if (!j.contentOffset) {
                        var f = $("#content"), r, q;
                        if (DM_Dir.start === "left") {
                            r = f.offset().left, q = h.windowWidth - 970 - g;
                            j.contentOffset = g + q / 2 - r
                        } else {
                            r = h.windowWidth - f.offset().left + f.width(), q = h.windowWidth - 970 - g;
                            j.contentOffset = q / 2*-1
                        }
                        if (e) {
                            var p = h.windowWidth - e.width;
                            j.bgPercentage = 50 + parseInt(j.contentOffset * 100 / p, 10)
                        }
                    }
                    if (b && j.bgPositionX) {
                        d.animate({
                            left: j.contentOffset
                        }, b);
                        e && a.animate({
                            "background-position-x": j.bgPercentage + "%"
                        }, b)
                    } else {
                        d.css("left", j.contentOffset);
                        e && a.css("background-position", j.bgPercentage + "% " + e.positionY)
                    }
                } else {
                    if (b && j.bgPositionX) {
                        d.animate({
                            left: 0
                        }, b);
                        e && a.animate({
                            "background-position-x": "50%"
                        }, b)
                    } else {
                        d.css("left", 0);
                        e && a.css("background-position", "50% " + e.positionY)
                    }
                }
            })
        },
        resizeNavigationPanelHeight: function(f) {
            var q = $(".sd_header"), b = q.offset().top + q.height(), e = $(window), r = e.height(), d = e.scrollTop(), a = r - b, c = d > b ? 0: (d < 0 ? b : b - d), p = d > b ? r: r - c;
            if (f) {
                c += 3;
                p -= 3
            }
            if (j.navHeight !== p || j.navTop !== c) {
                h.navigationPanel.css({
                    height: p,
                    top: c
                });
                j.navHeight = p;
                j.navTop = c
            }
        },
        toggleNavigationPanel: function() {
            return h.navigationPanelVisible ? h.closeNavigationPanel() : h.openNavigationPanel()
        },
        resizeResponsiveMenu: function() {
            var a = $(".container-fluid").not("#wrapper").height();
            h.navigationPanel.css("height", a).parent().css("height", a);
            h.navigationPanel.find("nav").height($(window).height() + $(".sd_header").height());
            if ($(window).width() > (1360 + 280 + ($(window).width() - 1360) / 2)) {
                $("#content .container-fluid").css("float", "none")
            } else {
                $("#content .container-fluid").css("float", "left")
            }
        },
        openNavigationPanel: function(a) {
            if (h.responsive) {
                $("#responsive-nav-panel").show();
                h.navigationPanelVisible = true;
                $("#wrapper").addClass("navigation-open");
                var b = $("body.skyscraper").length ? "calc(100% - 440px)": "calc(100% - 280px)";
                $("#content .container-fluid").css("width", b);
                h.navigationWrapper.width(280).height($("#content").height());
                $(".navigation-panel").css(DM_Dir.start, "0");
                h.resizeResponsiveMenu();
                setTimeout(function() {
                    $(".navigation-border").css(DM_Dir.end, "20px").show();
                    $("body").removeClass("no-transition");
                    PubSub.publish("player:resized");
                    h.headerNavigationTrigger.addClass("active")
                }, 300)
            } else {
                h.resizeNavigationPanelHeight();
                a = typeof a !== "undefined" ? a : 100;
                if (a) {
                    h.repositionContent(a, true);
                    h.navigationPanel.fadeIn(a, function() {
                        h.navigationPanelVisible = true
                    })
                } else {
                    h.navigationPanel.show();
                    h.navigationPanelVisible = true;
                    h.repositionContent(a)
                }
                h.headerNavigationTrigger.addClass("active")
            }
            h.nav.scrollTop(0);
            h.headerNavigationTrigger.addClass("active");
            store.set("navigation:status", "open");
            DM_Stats.trigger({
                action: "navigation_panel",
                context: "open"
            });
            h.store.set("nav-hide-notif", true);
            $("#header-navigation-trigger .navigation-notification").remove()
        },
        closeNavigationPanel: function() {
            if (h.responsive) {
                h.navigationPanelVisible = false;
                $("#wrapper").removeClass("navigation-open");
                var b = $("body.skyscraper").length ? ($(window).width() > 1200 ? "calc(100% - 160px)" : "100%"): "100%";
                $("#content .container-fluid").css("width", b);
                $("#content .container-fluid .row:first").removeAttr("style");
                h.setCloseResponsiveAttributes();
                var a = $("#player_container");
                if (a.length) {
                    if (a.width() / $(".js-player-box").width() >= 0.8) {
                        a.removeAttr("style")
                    }
                    setTimeout(function() {
                        a.removeAttr("style");
                        PubSub.publish("player:resizing");
                        PubSub.publish("player:resized")
                    }, 270)
                }
            } else {
                h.navigationPanel.fadeOut(100);
                h.navigationPanelVisible = false;
                h.repositionContent(100)
            }
            h.headerNavigationTrigger.removeClass("active");
            store.set("navigation:status", "closed");
            DM_Stats.trigger({
                action: "navigation_panel",
                context: "close"
            })
        },
        setCloseResponsiveAttributes: function() {
            $(".navigation-panel").css(DM_Dir.start, "calc(-" + g + "px - 10%)");
            h.navigationWrapper.width(0);
            $(".navigation-border").css(DM_Dir.end, "270px").hide();
            setTimeout(function() {
                $(".navigation-border").css(DM_Dir.end, "10px")
            }, 300)
        },
        toggleSubMenu: function(a, c) {
            a && a.preventDefault();
            c = typeof c !== "undefined" ? c : true;
            var b = $(this);
            b.closest("section").find(".nav-sub-menu")[c ? "slideToggle": "toggle"](300);
            if (c) {
                b.toggleClass("selected")
            } else {
                b.addClass("no-anim").toggleClass("selected");
                setTimeout(function() {
                    b.removeClass("no-anim")
                }, 300)
            }
            if (b.closest("section").data("content")) {
                h.blockStatus[b.closest("section").data("content")] = b.hasClass("selected") ? true : false;
                store.set(h.blockStoreKey, h.blockStatus)
            }
        },
        updateWatchLater: function(d) {
            var e = $("[data-watch-later-count]"), c = parseInt(e.data("count"), 10), a = d ? c + 1: c - 1, b = a > 10 ? "9+": a;
            e.text(b).data("count", a);
            return a ? e.closest("li").fadeIn() : e.closest("li").fadeOut()
        },
        loadDatas: function() {
            if (h.store.get("nav-datas") && h.store.get("nav-log-status") === DM_Context.is_logged) {
                try {
                    h.displayDatas(h.store.get("nav-datas"))
                } catch (a) {
                    h.loadRemoteDatas()
                }
            } else {
                if (h.store.get("nav-log-status") !== DM_Context.is_logged) {
                    h.clearDatas()
                } else {
                    h.store.remove("nav-datas");
                    h.store.remove("nav-subscriptions")
                }
                h.loadRemoteDatas()
            }
        },
        clearDatas: function() {
            h.store.remove("nav-datas");
            h.store.remove("nav-datas-subscribed-users");
            h.store.remove("nav-subscriptions");
            h.store.remove("nav-show-subscriptions");
            store.remove(h.blockStoreKey)
        },
        loadRemoteDatas: function() {
            DM_Controller.call("Shared_NavigationPanel", {
                method: "loadInitialDatas",
                callback: function(a) {
                    a = JSON.parse(a);
                    h.store.set("nav-datas-subscribed-users", a.subscribedUsers);
                    delete a.subscribedUsers;
                    h.store.set("nav-datas", a);
                    h.store.set("nav-log-status", DM_Context.is_logged);
                    h.displayDatas(a)
                }
            })
        },
        displayDatas: function(a) {
            k(a.recommendedUsers, $("#navigation-suggested-users"));
            k(a.channels, $("#navigation-channels-list"), true);
            if (DM_Context.is_logged) {
                h.displayInitialSubscriptions();
                h.displayExistingItems(a.userDatas)
            }
        },
        resetUserActivity: function(a) {
            var d = /^\/user\/([\S]+)\/[0-9]+/i, b = location.pathname.match(/^\/user\/([\S]+)\/[0-9]+/i), e = Array.isArray(b) && typeof b[1] ? b[1]: false, c = e && DM_Context.reader_login === e;
            if (!e || c) {
                return a
            }
            if (a) {
                for (var f in a.subscriptions) {
                    if (a.subscriptions[f].uri.substr(1) === e) {
                        if (a.subscriptions[f].activity !== 0) {
                            a.notifications -= a.subscriptions[f].activity;
                            a.subscriptions[f].activity = 0
                        }
                    }
                }
            }
            return a
        },
        displayInitialSubscriptions: function() {
            var a = h.store.get("nav-datas-subscribed-users");
            a = h.resetUserActivity(a);
            h.store.set("nav-datas-subscribed-users", a);
            h.displaySubscriptions(a);
            if (h.store.get("nav-show-subscriptions")) {
                setTimeout(function() {
                    h.loadAdditionnalSubscriptions(function(b) {
                        b = h.resetUserActivity(b);
                        h.store.set("nav-subscriptions", b);
                        h.displaySubscriptions(b, false)
                    });
                    h.subscriptionLoadMore.removeClass("loading load-more-button").addClass("loaded more-link");
                    h.subscriptionsLoaded = true
                }, 0)
            }
        },
        loadSubscriptions: function(b, d, c, a) {
            d = d || false;
            c = c || false;
            a = a || false;
            if (d&&!h.subscriptionLoadMore.hasClass("loaded")) {
                b && b.preventDefault()
            }
            if (h.subscriptionsLoaded) {
                return 
            }
            if (c) {
                h.displaySubscriptions(c, a)
            } else {
                h.loadAdditionnalSubscriptions(function(e) {
                    h.displaySubscriptions(e, a);
                    h.subscriptionLoadMore.removeClass("loading load-more-button").addClass("loaded more-link");
                    h.subscriptionsLoaded = true
                })
            }
        },
        loadAdditionnalSubscriptions: function(b) {
            var a = h.store.get("nav-subscriptions");
            if (a) {
                b(a)
            } else {
                DM_Controller.call("Shared_NavigationPanel", {
                    method: "getSubscribedUsers",
                    args: [h.subscriptionsPage, 5],
                    callback: function(c) {
                        c = $.parseJSON(c);
                        h.store.set("nav-subscriptions", c);
                        b(c)
                    }
                })
            }
        },
        displaySubscriptions: function(a) {
            h.subscriptionsPage++;
            if (typeof a.subscriptions === "undefined" ||!a.subscriptions.length) {
                return 
            }
            var b = $(".navigation-notification"), c = parseInt(b.data("total"), 10), d = c + parseInt(a.notifications, 10);
            b.data("total", d);
            d = d < 99 ? d : "99+";
            if (d) {
                b.text(d).removeClass("hidden")
            }
            k(a.subscriptions, $("#navigation-subscription-list"));
            h.subscriptionsPage > 2 && h.subscriptionLoadMore.addClass("loaded")
        },
        displayExistingItems: function(b) {
            var a = $("#user-infos-section");
            for (var c in b) {
                var d = parseInt(b[c], 10);
                d && a.find("[data-id=" + c + "]").show()
            }
        },
        highlightCurrentLink: function() {
            h.navigationPanel.find("ul a").each(function() {
                var a = $(this);
                if (a.attr("href") == location.pathname || a.attr("href") == location.href) {
                    a.addClass("current");
                    var b = a.closest("section").find(".navigation-sub-trigger");
                    b.hasClass("selected") || h.toggleSubMenu.call(b, false);
                    return false
                }
            })
        },
        isAutoOpenRoute: function() {
            var a = location.pathname.split("/"), b = ["admin", "profile"];
            return typeof a[1] !== "undefined" && b.indexOf(a[1])===-1
        },
        __localstorageConf: {
            "nav-datas": {
                version: 0.1,
                expire: 86400000
            },
            "nav-datas-subscribed-users": {
                version: 0.1,
                expire: 86400000
            },
            "nav-subscriptions": {
                version: 0.1,
                expire: 86400000
            },
            "nav-show-subscriptions": {
                version: 0.1,
                expire: 86400000
            },
            "nav-log-status": {
                expire: 86400000
            },
            "nav-hide-notif": {
                expire: 43200000
            }
        },
        __initialize: function() {
            $("#navigation-panel").length && h.initNavigationPanel()
        }
    };
    return h
}, true);
BOOMR_start = new Date().getTime();
(function(g) {
    if (g.parent !== g && document.getElementById("boomr-if-as") && document.getElementById("boomr-if-as").nodeName.toLowerCase() === "script") {
        g = g.parent
    }
    var i, j, d, h = g.document;
    if (g.BOOMR === undefined) {
        g.BOOMR = {}
    }
    BOOMR = g.BOOMR;
    if (BOOMR.version) {
        return 
    }
    BOOMR.version = "0.9";
    BOOMR.window = g;
    i = {
        beacon_url: "",
        site_domain: g.location.hostname.replace(/.*?([^.]+\.[^.]+)\.?$/, "$1").toLowerCase(),
        user_ip: "",
        onloadfired: false,
        handlers_attached: false,
        events: {
            page_ready: [],
            page_unload: [],
            dom_loaded: [],
            visibility_changed: [],
            before_beacon: [],
            click: []
        },
        vars: {},
        disabled_plugins: {},
        onclick_handler: function(b) {
            var a;
            if (!b) {
                b = g.event
            }
            if (b.target) {
                a = b.target
            } else {
                if (b.srcElement) {
                    a = b.srcElement
                }
            }
            if (a.nodeType === 3) {
                a = a.parentNode
            }
            if (a && a.nodeName.toUpperCase() === "OBJECT" && a.type === "application/x-shockwave-flash") {
                return 
            }
            i.fireEvent("click", a)
        },
        fireEvent: function(f, b) {
            var e, c, a;
            if (!this.events.hasOwnProperty(f)) {
                return false
            }
            a = this.events[f];
            for (e = 0; e < a.length; e++) {
                c = a[e];
                c[0].call(c[2], b, c[1])
            }
            return true
        },
        addListener: function(b, a, c) {
            if (b.addEventListener) {
                b.addEventListener(a, c, false)
            } else {
                if (b.attachEvent) {
                    b.attachEvent("on" + a, c)
                }
            }
        }
    };
    j = {
        t_start: BOOMR_start,
        t_end: null,
        utils: {
            objectToString: function(a, b) {
                var c = [], e;
                if (!a || typeof a !== "object") {
                    return a
                }
                if (b === undefined) {
                    b = "\n\t"
                }
                for (e in a) {
                    if (Object.prototype.hasOwnProperty.call(a, e)) {
                        c.push(encodeURIComponent(e) + "=" + encodeURIComponent(a[e]))
                    }
                }
                return c.join(b)
            },
            getCookie: function(c) {
                if (!c) {
                    return null
                }
                c = " " + c + "=";
                var b, a;
                a = " " + h.cookie + ";";
                if ((b = a.indexOf(c)) >= 0) {
                    b += c.length;
                    a = a.substring(b, a.indexOf(";", b));
                    return a
                }
                return null
            },
            setCookie: function(f, e, n) {
                var c, m, a, b;
                if (!f ||!i.site_domain) {
                    return false
                }
                c = this.objectToString(e, "&");
                m = f + "=" + c;
                a = [m, "path=/", "domain=" + i.site_domain];
                if (n) {
                    b = new Date();
                    b.setTime(b.getTime() + n * 1000);
                    b = b.toGMTString();
                    a.push("expires=" + b)
                }
                if (m.length < 4000) {
                    h.cookie = a.join("; ");
                    return (c === this.getCookie(f))
                }
                return false
            },
            getSubCookies: function(c) {
                var e, f, l, a, b = {};
                if (!c) {
                    return null
                }
                e = c.split("&");
                if (e.length === 0) {
                    return null
                }
                for (f = 0, l = e.length; f < l; f++) {
                    a = e[f].split("=");
                    a.push("");
                    b[decodeURIComponent(a[0])] = decodeURIComponent(a[1])
                }
                return b
            },
            removeCookie: function(a) {
                return this.setCookie(a, {}, 0)
            },
            pluginConfig: function(a, m, c, e) {
                var f, b = 0;
                if (!m ||!m[c]) {
                    return false
                }
                for (f = 0; f < e.length; f++) {
                    if (m[c][e[f]] !== undefined) {
                        a[e[f]] = m[c][e[f]];
                        b++
                    }
                }
                return (b > 0)
            }
        },
        init: function(c) {
            var a, e, b = ["beacon_url", "site_domain", "user_ip"];
            if (!c) {
                c = {}
            }
            for (a = 0; a < b.length; a++) {
                if (c[b[a]] !== undefined) {
                    i[b[a]] = c[b[a]]
                }
            }
            if (c.log !== undefined) {
                this.log = c.log
            }
            if (!this.log) {
                this.log = function() {}
            }
            for (e in this.plugins) {
                if (this.plugins.hasOwnProperty(e)) {
                    if (c[e] && c[e].hasOwnProperty("enabled") && c[e].enabled === false) {
                        i.disabled_plugins[e] = 1;
                        continue
                    } else {
                        if (i.disabled_plugins[e]) {
                            delete i.disabled_plugins[e]
                        }
                    }
                    if (typeof this.plugins[e].init === "function") {
                        this.plugins[e].init(c)
                    }
                }
            }
            if (i.handlers_attached) {
                return this
            }
            if (!i.onloadfired && (c.autorun === undefined || c.autorun !== false)) {
                if (h.readyState && h.readyState === "complete") {
                    this.setImmediate(BOOMR.page_ready, null, null, BOOMR)
                } else {
                    if (g.onpagehide || g.onpagehide === null) {
                        i.addListener(g, "pageshow", BOOMR.page_ready)
                    } else {
                        i.addListener(g, "load", BOOMR.page_ready)
                    }
                }
            }
            i.addListener(g, "DOMContentLoaded", function() {
                i.fireEvent("dom_loaded")
            });
            (function() {
                var f = function() {
                    i.fireEvent("visibility_changed")
                };
                if (h.webkitVisibilityState) {
                    i.addListener(h, "webkitvisibilitychange", f)
                } else {
                    if (h.msVisibilityState) {
                        i.addListener(h, "msvisibilitychange", f)
                    } else {
                        if (h.visibilityState) {
                            i.addListener(h, "visibilitychange", f)
                        }
                    }
                }
                i.addListener(h, "mouseup", i.onclick_handler);
                if (!g.onpagehide && g.onpagehide !== null) {
                    i.addListener(g, "unload", function() {
                        BOOMR.window = g = null
                    })
                }
            }());
            i.handlers_attached = true;
            return this
        },
        page_ready: function() {
            if (i.onloadfired) {
                return this
            }
            i.fireEvent("page_ready");
            i.onloadfired = true;
            return this
        },
        setImmediate: function(c, b, e, a) {
            var f = function() {
                c.call(a || null, b, e || {});
                f = null
            };
            if (g.setImmediate) {
                g.setImmediate(f)
            } else {
                if (g.msSetImmediate) {
                    g.msSetImmediate(f)
                } else {
                    if (g.webkitSetImmediate) {
                        g.webkitSetImmediate(f)
                    } else {
                        if (g.mozSetImmediate) {
                            g.mozSetImmediate(f)
                        } else {
                            setTimeout(f, 10)
                        }
                    }
                }
            }
        },
        subscribe: function(q, a, f, p) {
            var e, b, r, c;
            if (!i.events.hasOwnProperty(q)) {
                return this
            }
            r = i.events[q];
            for (e = 0; e < r.length; e++) {
                b = r[e];
                if (b[0] === a && b[1] === f && b[2] === p) {
                    return this
                }
            }
            r.push([a, f || {}, p || null]);
            if (q === "page_ready" && i.onloadfired) {
                this.setImmediate(a, null, f, p)
            }
            if (q === "page_unload") {
                c = function(k) {
                    if (a) {
                        a.call(p, k || g.event, f)
                    }
                };
                if (g.onpagehide || g.onpagehide === null) {
                    i.addListener(g, "pagehide", c)
                } else {
                    i.addListener(g, "unload", c)
                }
                i.addListener(g, "beforeunload", c)
            }
            return this
        },
        addVar: function(c, b) {
            if (typeof c === "string") {
                i.vars[c] = b
            } else {
                if (typeof c === "object") {
                    var a = c, e;
                    for (e in a) {
                        if (a.hasOwnProperty(e)) {
                            i.vars[e] = a[e]
                        }
                    }
                }
            }
            return this
        },
        removeVar: function(b) {
            var c, a;
            if (!arguments.length) {
                return this
            }
            if (arguments.length === 1 && Object.prototype.toString.apply(b) === "[object Array]") {
                a = b
            } else {
                a = arguments
            }
            for (c = 0; c < a.length; c++) {
                if (i.vars.hasOwnProperty(a[c])) {
                    delete i.vars[a[c]]
                }
            }
            return this
        },
        sendBeacon: function() {
            var b, a, c, e = 0;
            for (b in this.plugins) {
                if (this.plugins.hasOwnProperty(b)) {
                    if (i.disabled_plugins[b]) {
                        continue
                    }
                    if (!this.plugins[b].is_complete()) {
                        return this
                    }
                }
            }
            i.vars.v = BOOMR.version;
            i.vars.u = h.URL.replace(/#.*/, "");
            if (g !== window) {
                i.vars["if"] = ""
            }
            i.fireEvent("before_beacon", i.vars);
            if (!i.beacon_url) {
                return this
            }
            a = [];
            for (b in i.vars) {
                if (i.vars.hasOwnProperty(b)) {
                    e++;
                    a.push(encodeURIComponent(b) + "=" + (i.vars[b] === undefined || i.vars[b] === null ? "" : encodeURIComponent(i.vars[b])))
                }
            }
            a = i.beacon_url + ((i.beacon_url.indexOf("?")>-1) ? "&" : "?") + a.join("&");
            BOOMR.debug("Sending url: " + a.replace(/&/g, "\n\t"));
            if (e) {
                c = new Image();
                c.src = a
            }
            return this
        }
    };
    delete BOOMR_start;
    (function() {
        var a = function(b) {
            return function(e, c) {
                this.log(e, b, "boomerang" + (c ? "." + c : ""));
                return this
            }
        };
        j.debug = a("debug");
        j.info = a("info");
        j.warn = a("warn");
        j.error = a("error")
    }());
    if (g.YAHOO && g.YAHOO.widget && g.YAHOO.widget.Logger) {
        j.log = g.YAHOO.log
    } else {
        if (g.Y && g.Y.log) {
            j.log = g.Y.log
        } else {
            if (typeof console === "object" && console.log !== undefined) {
                j.log = function(c, b, a) {
                    console.log(a + ": [" + b + "] " + c)
                }
            }
        }
    }
    for (d in j) {
        if (j.hasOwnProperty(d)) {
            BOOMR[d] = j[d]
        }
    }
    BOOMR.plugins = BOOMR.plugins || {}
}(window));
(function() {
    BOOMR = BOOMR || {};
    BOOMR.plugins = BOOMR.plugins || {};
    var b = {
        complete: false,
        done: function() {
            var a = BOOMR.window, g, j, h, i;
            g = a.performance || a.msPerformance || a.webkitPerformance || a.mozPerformance;
            if (g && g.timing && g.navigation) {
                BOOMR.info("This user agent supports NavigationTiming.", "nt");
                j = g.navigation;
                h = g.timing;
                i = {
                    nt_red_cnt: j.redirectCount,
                    nt_nav_type: j.type,
                    nt_nav_st: h.navigationStart,
                    nt_red_st: h.redirectStart,
                    nt_red_end: h.redirectEnd,
                    nt_fet_st: h.fetchStart,
                    nt_dns_st: h.domainLookupStart,
                    nt_dns_end: h.domainLookupEnd,
                    nt_con_st: h.connectStart,
                    nt_con_end: h.connectEnd,
                    nt_req_st: h.requestStart,
                    nt_res_st: h.responseStart,
                    nt_res_end: h.responseEnd,
                    nt_domloading: h.domLoading,
                    nt_domint: h.domInteractive,
                    nt_domcontloaded_st: h.domContentLoadedEventStart,
                    nt_domcontloaded_end: h.domContentLoadedEventEnd,
                    nt_domcomp: h.domComplete,
                    nt_load_st: h.loadEventStart,
                    nt_load_end: h.loadEventEnd,
                    nt_unload_st: h.unloadEventStart,
                    nt_unload_end: h.unloadEventEnd
                };
                if (h.secureConnectionStart) {
                    i.nt_ssl_st = h.secureConnectionStart
                }
                if (h.msFirstPaint) {
                    i.nt_first_paint = h.msFirstPaint
                }
                BOOMR.addVar(i)
            }
            if (a.chrome && a.chrome.loadTimes) {
                h = a.chrome.loadTimes();
                if (h) {
                    i = {
                        nt_spdy: (h.wasFetchedViaSpdy ? 1 : 0),
                        nt_first_paint: h.firstPaintTime
                    };
                    BOOMR.addVar(i)
                }
            }
            this.complete = true;
            BOOMR.sendBeacon()
        }
    };
    BOOMR.plugins.NavigationTiming = {
        init: function() {
            BOOMR.subscribe("page_ready", b.done, null, b);
            return this
        },
        is_complete: function() {
            return b.complete
        }
    }
}());
(function(e) {
    var f = e.document, d;
    BOOMR = BOOMR || {};
    BOOMR.plugins = BOOMR.plugins || {};
    d = {
        initialized: false,
        complete: false,
        timers: {},
        cookie: "RT",
        cookie_exp: 600,
        strict_referrer: true,
        navigationType: 0,
        navigationStart: undefined,
        responseStart: undefined,
        t_start: undefined,
        t_fb_approx: undefined,
        r: undefined,
        r2: undefined,
        setCookie: function(a, i) {
            var b, j = new Date().getTime(), c;
            if (!this.cookie) {
                return this
            }
            c = BOOMR.utils.getSubCookies(BOOMR.utils.getCookie(this.cookie)) || {};
            if (a) {
                c[a] = j
            }
            c.r = f.URL.replace(/#.*/, "");
            if (a === "cl") {
                if (i) {
                    c.nu = i
                } else {
                    if (c.nu) {
                        delete c.nu
                    }
                }
            }
            if (i === false) {
                delete c.nu
            }
            BOOMR.debug("Setting cookie " + BOOMR.utils.objectToString(c), "rt");
            if (!BOOMR.utils.setCookie(this.cookie, c, this.cookie_exp)) {
                BOOMR.error("cannot set start cookie", "rt");
                return this
            }
            b = new Date().getTime();
            if (b - j > 50) {
                BOOMR.utils.removeCookie(this.cookie);
                BOOMR.error("took more than 50ms to set cookie... aborting: " + j + " -> " + b, "rt")
            }
            return this
        },
        initFromCookie: function() {
            var a;
            if (!this.cookie) {
                return 
            }
            a = BOOMR.utils.getSubCookies(BOOMR.utils.getCookie(this.cookie));
            if (!a) {
                return 
            }
            a.s = Math.max( + a.ul || 0, + a.cl || 0);
            BOOMR.debug("Read from cookie " + BOOMR.utils.objectToString(a), "rt");
            if (a.s && a.r) {
                this.r = a.r;
                if (!this.strict_referrer || this.r === this.r2 || (a.s ===+ a.cl && a.nu === f.URL.replace(/#.*/, ""))) {
                    this.t_start = a.s;
                    if ( + a.hd > a.s) {
                        this.t_fb_approx = parseInt(a.hd, 10)
                    }
                } else {
                    this.t_start = this.t_fb_approx = undefined
                }
            }
        },
        checkPreRender: function() {
            if (!(f.webkitVisibilityState && f.webkitVisibilityState === "prerender")&&!(f.msVisibilityState && f.msVisibilityState === 3)) {
                return false
            }
            BOOMR.plugins.RT.startTimer("t_load", this.navigationStart);
            BOOMR.plugins.RT.endTimer("t_load");
            BOOMR.plugins.RT.startTimer("t_prerender", this.navigationStart);
            BOOMR.plugins.RT.startTimer("t_postrender");
            BOOMR.subscribe("visibility_changed", BOOMR.plugins.RT.done, null, BOOMR.plugins.RT);
            return true
        },
        initNavTiming: function() {
            var c, a, b;
            if (this.navigationStart) {
                return 
            }
            a = e.performance || e.msPerformance || e.webkitPerformance || e.mozPerformance;
            if (a && a.navigation) {
                this.navigationType = a.navigation.type
            }
            if (a && a.timing) {
                c = a.timing
            } else {
                if (e.chrome && e.chrome.csi && e.chrome.csi().startE) {
                    c = {
                        navigationStart: e.chrome.csi().startE
                    };
                    b = "csi"
                } else {
                    if (e.gtbExternal && e.gtbExternal.startE()) {
                        c = {
                            navigationStart: e.gtbExternal.startE()
                        };
                        b = "gtb"
                    }
                }
            }
            if (c) {
                BOOMR.addVar("rt.start", b || "navigation");
                this.navigationStart = c.navigationStart || c.fetchStart || undefined;
                this.responseStart = c.responseStart || undefined;
                if (navigator.userAgent.match(/Firefox\/[78]\./)) {
                    this.navigationStart = c.unloadEventStart || c.fetchStart || undefined
                }
            } else {
                BOOMR.warn("This browser doesn't support the WebTiming API", "rt")
            }
            return 
        },
        page_unload: function(a) {
            BOOMR.debug("Unload called with " + BOOMR.utils.objectToString(a), "rt");
            this.setCookie(a.type === "beforeunload" ? "ul" : "hd")
        },
        onclick: function(a) {
            if (!a) {
                return 
            }
            BOOMR.debug("Click called with " + a.nodeName, "rt");
            while (a && a.nodeName.toUpperCase() !== "A") {
                a = a.parentNode
            }
            if (a && a.nodeName.toUpperCase() === "A") {
                BOOMR.debug("passing through", "rt");
                this.setCookie("cl", a.href)
            }
        },
        domloaded: function() {
            BOOMR.plugins.RT.endTimer("t_domloaded")
        }
    };
    BOOMR.plugins.RT = {
        init: function(a) {
            BOOMR.debug("init RT", "rt");
            if (e !== BOOMR.window) {
                e = BOOMR.window;
                f = e.document
            }
            BOOMR.utils.pluginConfig(d, a, "RT", ["cookie", "cookie_exp", "strict_referrer"]);
            d.initFromCookie();
            if (d.initialized) {
                return this
            }
            d.complete = false;
            d.timers = {};
            BOOMR.subscribe("page_ready", this.done, null, this);
            BOOMR.subscribe("dom_loaded", d.domloaded, null, d);
            BOOMR.subscribe("page_unload", d.page_unload, null, d);
            BOOMR.subscribe("click", d.onclick, null, d);
            if (BOOMR.t_start) {
                this.startTimer("boomerang", BOOMR.t_start);
                this.endTimer("boomerang", BOOMR.t_end);
                this.endTimer("boomr_fb", BOOMR.t_start)
            }
            d.r = d.r2 = f.referrer.replace(/#.*/, "");
            d.initialized = true;
            return this
        },
        startTimer: function(b, a) {
            if (b) {
                if (b === "t_page") {
                    this.endTimer("t_resp", a)
                }
                d.timers[b] = {
                    start: (typeof a === "number" ? a : new Date().getTime())
                };
                d.complete = false
            }
            return this
        },
        endTimer: function(b, a) {
            if (b) {
                d.timers[b] = d.timers[b] || {};
                if (d.timers[b].end === undefined) {
                    d.timers[b].end = (typeof a === "number" ? a : new Date().getTime())
                }
            }
            return this
        },
        setTimer: function(b, a) {
            if (b) {
                d.timers[b] = {
                    delta: a
                }
            }
            return this
        },
        done: function() {
            BOOMR.debug("Called done", "rt");
            var m, k = new Date().getTime(), b = {
                t_done: 1,
                t_resp: 1,
                t_page: 1
            }, n = 0, c, a, l = [];
            d.complete = false;
            d.initFromCookie();
            d.initNavTiming();
            if (d.checkPreRender()) {
                return this
            }
            if (d.responseStart) {
                this.endTimer("t_resp", d.responseStart);
                if (d.timers.t_load) {
                    this.setTimer("t_page", d.timers.t_load.end - d.responseStart)
                } else {
                    this.setTimer("t_page", k - d.responseStart)
                }
            } else {
                if (d.timers.hasOwnProperty("t_page")) {
                    this.endTimer("t_page")
                } else {
                    if (d.t_fb_approx) {
                        this.endTimer("t_resp", d.t_fb_approx);
                        this.setTimer("t_page", k - d.t_fb_approx)
                    }
                }
            }
            if (d.timers.hasOwnProperty("t_postrender")) {
                this.endTimer("t_postrender");
                this.endTimer("t_prerender")
            }
            if (d.navigationStart) {
                m = d.navigationStart
            } else {
                if (d.t_start && d.navigationType !== 2) {
                    m = d.t_start;
                    BOOMR.addVar("rt.start", "cookie")
                } else {
                    BOOMR.addVar("rt.start", "none");
                    m = undefined
                }
            }
            this.endTimer("t_done", k);
            BOOMR.removeVar("t_done", "t_page", "t_resp", "r", "r2", "rt.tstart", "rt.bstart", "rt.end", "t_postrender", "t_prerender", "t_load");
            BOOMR.addVar("rt.tstart", m);
            BOOMR.addVar("rt.bstart", BOOMR.t_start);
            BOOMR.addVar("rt.end", d.timers.t_done.end);
            for (c in d.timers) {
                if (d.timers.hasOwnProperty(c)) {
                    a = d.timers[c];
                    if (typeof a.delta !== "number") {
                        if (typeof a.start !== "number") {
                            a.start = m
                        }
                        a.delta = a.end - a.start
                    }
                    if (isNaN(a.delta)) {
                        continue
                    }
                    if (b.hasOwnProperty(c)) {
                        BOOMR.addVar(c, a.delta)
                    } else {
                        l.push(c + "|" + a.delta)
                    }
                    n++
                }
            }
            if (n) {
                BOOMR.addVar("r", d.r);
                if (d.r2 !== d.r) {
                    BOOMR.addVar("r2", d.r2)
                }
                if (l.length) {
                    BOOMR.addVar("t_other", l.join(","))
                }
            }
            d.timers = {};
            d.complete = true;
            BOOMR.sendBeacon();
            return this
        },
        is_complete: function() {
            return d.complete
        }
    }
}(window));
_DM.define("DM_BOOMR_Init", function() {
    var b = {
        promises: {},
        enabled: function() {
            return window.BOOMREnabled && (!window.BOOMRWaitForPlayer || (!window.videoHasPartnerPlayer && window.flashOK === true))
        },
        sendBeacon: function(e) {
            var a = true;
            b.promises[e] = true;
            for (var f in b.promises) {
                if (!b.promises.hasOwnProperty(f)) {
                    continue
                }
                a = a && b.promises[f]
            }
            if (a) {
                BOOMR.page_ready()
            }
        },
        onBeforeBeacon: function(f) {
            var h = ["u", "v", "nt_red_cnt", "nt_nav_type", "nt_red_st", "nt_red_end", "nt_spdy", "nt_dns_st", "nt_dns_end", "nt_con_st", "nt_con_end", "nt_unload_st", "nt_unload_end", "nt_domloading", "nt_domint", "nt_fet_st", "nt_load_st", "nt_domcomp", "r", "r2", "t_other", "rt.start", "rt.tstart", "rt.bstart", "rt.end", "nt_res_end"];
            for (var g = 0, a = h.length; g < a; g++) {
                BOOMR.removeVar(h[g])
            }
            if (f.nt_first_paint && (f.nt_first_paint + "").match(/\./)) {
                f.nt_first_paint = Math.ceil(f.nt_first_paint * 1000)
            }
            BOOMR.addVar("t_toppage", dm_times.start);
            BOOMR.addVar("t_first_domready", dm_times.firstdomready);
            PubSub.publish("boomerang:before_beacon", f)
        },
        setupForPlayer: function() {
            BOOMR.addVar("with_ad_call", window.auditudeEnabledForVideo);
            b.promises = {
                pageLoaded: false,
                playerPlaying: false
            };
            $(window).on("load", function() {
                setTimeout(function() {
                    b.sendBeacon("pageLoaded")
                }, 0)
            }).on("beforeunload", function() {
                BOOMR.addVar("t_usergone", new Date().getTime());
                BOOMR.page_ready()
            });
            PubSub.subscribe("boomerang:player_data", function(a) {
                BOOMR.addVar("cell", a.cell);
                delete (a.cell);
                for (var d in a) {
                    if (a.hasOwnProperty(d)) {
                        BOOMR.addVar(d, a[d])
                    }
                }
                b.sendBeacon("playerPlaying")
            })
        },
        addDefaultParameters: function() {
            navigator.cookieEnabled && BOOMR.addVar("cksz", document.cookie.length);
            BOOMR.addVar("browser", $.browser.name);
            BOOMR.addVar("country", window.DM_Context ? DM_Context.geoip_country : "unknown");
            BOOMR.addVar("routename", window.DM_Context ? DM_Context.route_name : "unknown");
            BOOMR.addVar("logged", window.DM_Context ? DM_Context.is_logged : "unknown");
            BOOMR.addVar("AS", window.DM_Context ? DM_Context.geoip_as : "unknown");
            window.bottomBOOMR = b.addBottomPageParameters()
        },
        addBottomPageParameters: function() {
            BOOMR.addVar("t_env_top", dm_times.env_top);
            BOOMR.addVar("t_script_bottom", dm_times.script_bottom)
        },
        __initialize: function() {
            if (!b.enabled()) {
                return 
            }
            BOOMR.init({
                beacon_url: window.BOOMRBeaconUrl,
                log: null,
                autorun: !window.BOOMRWaitForPlayer
            });
            b.addDefaultParameters();
            BOOMR.subscribe("before_beacon", b.onBeforeBeacon);
            if (window.BOOMRWaitForPlayer && window.DM_PlayerEvents) {
                b.setupForPlayer()
            }
        }
    };
    return b
});
(function(l) {
    function m(a, b) {
        return (typeof a == "function") ? (a.call(b)) : a
    }
    function j(a) {
        while (a = a.parentNode) {
            if (a == document) {
                return true
            }
        }
        return false
    }
    function i(b, c) {
        this.$element = l(b);
        this.options = c;
        this.options.close_button_set = false;
        if (this.$element.hasClass("js-tipsy-ballon") || this.$element.hasClass("js-tipsy-infotip")) {
            this.options.tooltip_type = (this.$element.hasClass("js-tipsy-ballon") ? "tipsy-ballon" : "tipsy-infotip");
            this.options.delayOut = 50;
            this.options.delayIn = 100;
            this.options.interactive = (this.$element.hasClass("js-tipsy-ballon") ? false : true);
            this.options.tooltip_type = (this.$element.hasClass("js-tipsy-ballon") ? "tipsy-ballon" : "tipsy-infotip");
            this.options.gravity = "s";
            this.options.html = true;
            this.options.fade = true
        }
        if (this.options.tooltip_type == "tipsy-ballon") {
            this.options.delayIn = 400;
            this.options.offset = 1;
            this.options.fade = true;
            this.options.delayOut = 150
        }
        if (this.options.tooltip_type == "tipsy-infotip") {
            this.options.offset = 3
        }
        var a = this.$element.attr("class").match(/js-tipsy-gravity-(\w+)/);
        if (a) {
            this.options.gravity = a[1]
        }
        if (this.$element.hasClass("js-tipsy-form-help")) {
            this.options.tooltip_type += " tipsy-form-help"
        }
        if (this.$element.hasClass("js-tipsy-preserve")) {
            this.options.preserve = true
        }
        this.enabled = true;
        this.fixTitle()
    }
    i.prototype = {
        show: function() {
            var c = this.getTitle();
            if (c && this.enabled) {
                var d = this.tip();
                if (d.is(":visible")) {
                    return true
                }
                if (this.options.preserve) {
                    l(this.options.target).children().detach().prependTo(d.find(".tipsy-inner"))
                } else {
                    d.find(".tipsy-inner")[this.options.html ? "html": "text"](c)
                }
                d[0].className = "tipsy";
                if (this.options.tooltip_type) {
                    d.addClass(this.options.tooltip_type)
                }
                d.detach().css({
                    top: 0,
                    left: 0,
                    visibility: "hidden",
                    display: "block"
                }).prependTo(document.body);
                var g = l.extend({}, this.$element.offset(), {
                    width: this.$element[0].offsetWidth,
                    height: this.$element[0].offsetHeight
                });
                var f = d[0].offsetWidth, p = d[0].offsetHeight, a = m(this.options.gravity, this.$element[0]);
                var e;
                switch (a.charAt(0)) {
                case"n":
                    e = {
                        top: g.top + g.height + this.options.offset,
                        left: g.left + g.width / 2 - f / 2
                    };
                    break;
                case"s":
                    e = {
                        top: g.top - p - this.options.offset,
                        left: g.left + g.width / 2 - f / 2
                    };
                    break;
                case"e":
                    e = {
                        top: g.top + g.height / 2 - p / 2,
                        left: g.left - f - this.options.offset
                    };
                    break;
                case"w":
                    e = {
                        top: g.top + g.height / 2 - p / 2,
                        left: g.left + g.width + this.options.offset
                    };
                    break
                }
                if (a.length == 2) {
                    if (a.charAt(1) == "w") {
                        e.left = g.left - 4
                    } else {
                        e.left = g.left + g.width - f
                    }
                }
                d.css(e).addClass("tipsy-" + a);
                d.find(".tipsy-arrow")[0].className = "tipsy-arrow tipsy-arrow-" + a.charAt(0);
                if (this.options.className) {
                    d.addClass(m(this.options.className, this.$element[0]))
                }
                if (this.options.fade) {
                    d.stop().css({
                        opacity: 0,
                        display: "block",
                        visibility: "visible"
                    }).animate({
                        opacity: this.options.opacity
                    })
                } else {
                    d.css({
                        visibility: "visible",
                        opacity: this.options.opacity
                    })
                }
                if (this.options.close_button) {
                    _this = this;
                    d.find(".js-tipsy-close").on("click", function() {
                        _this.hide();
                        PubSub.publish("dm:tipsy:closed", d)
                    })
                }
                if (this.options.showCallback) {
                    var b = this.options.showCallback.split(".");
                    if (b.length === 2) {
                        b = window[b[0]][b[1]]
                    } else {
                        b = window[this.options.showCallback]
                    }
                    typeof b === "function" && b(this)
                }
            }
        },
        hide: function() {
            if (this.options.target && l(".tipsy-inner", this.tip()).html() !== "") {
                if (this.options.close_button&&!this.options.close_button_set) {
                    this.options.close_button_set = true
                }
                if (this.options.preserve) {
                    l(".tipsy-inner", this.tip()).children().detach().prependTo(this.options.target)
                } else {
                    l(this.options.target).html(l(".tipsy-inner", this.tip()).html())
                }
            }
            if (this.options.fade) {
                var a = this;
                this.tip().stop().fadeOut(function() {
                    a.hideCallback.call(a);
                    l(this).remove()
                })
            } else {
                this.tip().remove();
                this.hideCallback.call(this)
            }
        },
        hideCallback: function() {
            if (this.options.hideCallback) {
                var a = this.options.hideCallback.split(".");
                if (a.length === 2) {
                    a = window[a[0]][a[1]]
                } else {
                    a = window[this.options.hideCallback]
                }
                typeof a === "function" && a(this)
            }
        },
        fixTitle: function() {
            var a = this.$element;
            if (a.attr("title") || typeof(a.attr("original-title")) != "string") {
                a.attr("original-title", a.attr("title") || "").removeAttr("title")
            }
        },
        getTitle: function() {
            var a, d = this.$element, b = this.options;
            if (b.preserve) {
                return true
            }
            this.fixTitle();
            if (b.target) {
                a = l(b.target).hide().html()
            } else {
                if (typeof b.title == "string") {
                    a = d.attr(b.title == "title" ? "original-title" : b.title)
                } else {
                    if (typeof b.title == "function") {
                        a = b.title.call(d[0])
                    }
                }
                a = ("" + a).replace(/(^\s*|\s*$)/, "")
            }
            if (!this.options.close_button_set && this.options.tooltip_type == "tipsy-infotip" && this.options.close_button) {
                var c = '<a class="js-tipsy-close icon icon-close pull-end-abs mrg-end-md font-xsm foreground2 cliquable"></a>' + a;
                a = c
            }
            return a || b.fallback
        },
        tip: function() {
            if (!this.$tip) {
                this.$tip = l('<div class="tipsy"></div>').html('<div class="tipsy-inner"></div><div class="tipsy-arrow-bg"></div><div class="tipsy-arrow"></div>');
                this.$tip.data("tipsy-pointee", this.$element[0])
            }
            return this.$tip
        },
        validate: function() {
            if (!this.$element[0].parentNode) {
                this.hide();
                this.$element = null;
                this.options = null
            }
        },
        enable: function() {
            this.enabled = true
        },
        disable: function() {
            this.enabled = false
        },
        toggleEnabled: function() {
            this.enabled=!this.enabled
        }
    };
    function n(b, c) {
        var a = l.data(b, "tipsy");
        if (!a) {
            a = new i(b, l.fn.tipsy.elementOptions(b, c));
            l.data(b, "tipsy", a)
        }
        return a
    }
    function h(b) {
        var c = b.data.options, a = n(this, c);
        a.hoverState = "in";
        if (a.options.tooltip_type == "tipsy-ballon") {
            c.delayIn = 400
        }
        setTimeout(function() {
            if (a.hoverState == "in") {
                a.fixTitle();
                a.show();
                PubSub.publish("dm:tipsy:showed", a)
            }
        }, a.options.delayIn)
    }
    function k(c) {
        var d = c.data.options, b = n(this, d);
        b.hoverState = "out";
        if (d.delayOut === 0) {
            b.hide();
            PubSub.publish("dm:tipsy:hidden", b)
        } else {
            var e = setTimeout(function() {
                if (b.hoverState == "out") {
                    b.hide()
                }
                PubSub.publish("dm:tipsy:hidden", b)
            }, b.options.delayOut);
            if (d.interactive) {
                var a = b.tip();
                a.hover(function() {
                    clearTimeout(e)
                }, function() {
                    e = setTimeout(function() {
                        if (b.hoverState == "out") {
                            b.hide();
                            PubSub.publish("dm:tipsy:hidden", b)
                        }
                    }, d.delayOut)
                })
            }
        }
    }
    l.fn.tipsy = function(d) {
        if (d === true) {
            return this.data("tipsy")
        } else {
            if (typeof d == "string") {
                var b = this.data("tipsy");
                if (b) {
                    b[d]()
                }
                return this
            }
        }
        d = l.extend({}, l.fn.tipsy.defaults, d);
        if (d.target) {
            d.html = true;
            d.interactive = true;
            d.delayOut = 100
        }
        if (!d.live) {
            this.each(function() {
                n(this, d)
            })
        }
        if (d.trigger != "manual") {
            var c = d.trigger == "hover" ? "mouseenter": "focus", e = d.trigger == "hover" ? "mouseleave": "blur", a = {
                options: d
            };
            if (d.live) {
                l(this.context).on(c, a, h).on(e, a, k)
            } else {
                this.on(c, a, h).on(e, a, k)
            }
        }
        return this
    };
    l.fn.untipsy = function() {
        if (l(this).data("tipsy")) {
            var a = l(this).data("tipsy"), c = a.options, b = c.trigger == "hover" ? "mouseenter": "focus", d = c.trigger == "hover" ? "mouseleave": "blur";
            if (c.live) {
                l(this.context).off(b, h).off(d, k)
            } else {
                this.off(b, h).off(d, k)
            }
            if (c.interactive) {
                $tip = a.tip();
                $tip.off("hover")
            }
            a.hide();
            l(this).removeData("tipsy")
        }
        return this
    };
    l.fn.tipsy.defaults = {
        className: null,
        delayIn: 0,
        delayOut: 0,
        fade: false,
        fallback: "",
        gravity: "n",
        html: false,
        live: false,
        offset: 0,
        opacity: 1,
        title: "title",
        trigger: "hover",
        interactive: false,
        showCallback: false,
        hideCallback: false
    };
    l.fn.tipsy.revalidate = function() {
        l(".tipsy").each(function() {
            var a = l.data(this, "tipsy-pointee");
            if (!a ||!j(a)) {
                l(this).remove()
            }
        })
    };
    l.fn.tipsy.elementOptions = function(a, b) {
        return l.metadata ? l.extend({}, b, l(a).metadata()) : b
    };
    l.fn.tipsy.autoNS = function() {
        return l(this).offset().top > (l(document).scrollTop() + l(window).height() / 2) ? "s" : "n"
    };
    l.fn.tipsy.autoWE = function() {
        return l(this).offset().left > (l(document).scrollLeft() + l(window).width() / 2) ? "e" : "w"
    };
    l.fn.tipsy.autoBounds = function(a, b) {
        return function() {
            var f = {
                ns: b[0],
                ew: (b.length > 1 ? b[1] : false)
            }, c = l(document).scrollTop() + a, e = l(document).scrollLeft() + a, d = l(this);
            if (d.offset().top < c) {
                f.ns = "n"
            }
            if (d.offset().left < e) {
                f.ew = "w"
            }
            if (l(window).width() + l(document).scrollLeft() - d.offset().left < a) {
                f.ew = "e"
            }
            if (l(window).height() + l(document).scrollTop() - d.offset().top < a) {
                f.ns = "s"
            }
            return f.ns + (f.ew ? f.ew : "")
        }
    };
    l(function() {
        l(".js-tipsy").each(function() {
            var a = l(this).data("tipsy-options") || {};
            l(this).tipsy(a)
        })
    })
})(jQuery);
_DM.define("Sd_User_Subscribe", function() {
    var b = {
        updateButton: function(a, d) {
            $.ajax({
                url: "/controller/Shared_User_Subscribe",
                data: d,
                success: function(g, c, h) {
                    $(a).replaceWith(g)
                }
            })
        },
        updateButtons: function(a) {
            $('.sd_user_subscribe[data-user="' + a + '"]').each(function() {
                var d = {
                    request: "/" + a,
                    config: $(this).attr("data-config"),
                    subscribe_context: $(this).find("button").data("subscribe-context")
                };
                b.updateButton(this, d)
            })
        },
        setCount: function(j, a) {
            var i = $('[data-follower-count="' + j + '"]');
            var h = parseInt($('[data-follower-count="' + j + '"]').data("count"), 10);
            var g = h + a;
            i.data("count", g).html(g)
        },
        afterToggleSubscribe: function(h, a, g) {
            var f = h.action;
            PubSub.publish("dm:user_logged:" + f, a);
            if (!b.isSameOrigin() ||!h.needed_login) {
                b.updateButtons(a)
            }
            if (f == "subscribe") {
                if (g) {
                    if (g == "from-channel-tooltip") {
                        DM_Stats.trigger({
                            action: "follow",
                            context: "from_channel_tooltip"
                        })
                    } else {
                        if (g == "push-follow-toolip") {
                            DM_Stats.trigger({
                                action: "follow",
                                context: "with_push_tooltip"
                            })
                        }
                    }
                } else {
                    DM_Stats.trigger({
                        action: "follow"
                    })
                }
                b.setCount(a, 1)
            } else {
                DM_Stats.trigger({
                    action: "unfollow"
                });
                b.setCount(a, - 1)
            }
            Sd_Navigation_Panel.clearDatas()
        },
        isSameOrigin: function() {
            var a;
            try {
                a = top.location.href
            } catch (d) {
                a = false
            }
            return a
        },
        onClick: function(i) {
            var k = $(i.currentTarget), a = k.data("href"), l = k.parents(".sd_user_subscribe").data("css-class"), j = k.data("user-uri"), h = {
                request: j,
                next_url: k.data("next-href"),
                redirect_url: b.isSameOrigin() ? j: document.location.href
            };
            if (k.data("no-count")) {
                h.no_count = true
            }
            if (DM_Context.is_logged === false) {
                DM_Stats.trigger({
                    action: "follow",
                    context: "unlogged"
                })
            }
            DM_Ajax.jsonCall(a, h, function(d) {
                var c = $.parseJSON(d);
                b.afterToggleSubscribe(c, k.closest(".sd_user_subscribe").data("user"), k.data("subscribe-context"))
            }, function(c, d) {
                if (c == "too_many_subscriptions") {
                    alert(d)
                }
            })
        },
        initEvents: function() {
            $(document).on("click", ".sd_user_subscribe button", b.onClick)
        },
        loadLazyButtons: function() {
            var d = [], a = 0;
            $("[data-lazy-follow-button]").each(function() {
                var f = $(this).parent(), c = "lazy-" + a;
                f.attr("data-lazy", c);
                d.push({
                    user: f.data("user"),
                    config: f.data("config"),
                    id: c
                });
                a++
            });
            if (!d.length) {
                return 
            }
            DM_Ajax.jsonCall("/controller/Shared_User_Subscribe?method=loadButton", {
                buttons: JSON.stringify(d)
            }, function(f) {
                f = JSON.parse(f);
                for (var c in f) {
                    if (f[c].match(/icon-plus/)) {
                        f[c] = f[c].replace(/btn-sm/, "btn-3-states btn-default-to-success button btn-sm").replace(/btn-success/, "")
                    }
                    $("[data-lazy=" + c + "]").replaceWith(f[c])
                }
            }, function() {})
        },
        __onDomReady: function() {
            b.initEvents();
            b.loadLazyButtons()
        }
    };
    return b
}, true);
_DM.define("DM_ReaderSession", function() {
    function d(b) {
        var a = storeWithExpiration.get(e.storeKey)[b];
        if (a === void 0 || a === "null") {
            a = null
        }
        return a
    }
    function f(c, a) {
        var b = storeWithExpiration.get(e.storeKey), i = storeWithExpiration.getNewExp(e.storeKey), j = b || {};
        j[c] = a;
        storeWithExpiration.set(e.storeKey, j, i);
        return storeWithExpiration.get(e.storeKey)
    }
    var e = {
        storeKey: "DM_ReaderSession",
        add: function(b, a) {
            switch (b) {
            case"watchedVideo":
                e.addDefaultList(b, a, "watchedVideos");
                e.addVideoByLogin("watchedVideosByLogin", a);
                PubSub.publish("dm:readersession:video_watched", a);
                break;
            case"startedVideo":
                e.addDefaultList(b, a, "startedVideos");
                e.addVideoByLogin("startedVideosByLogin", a);
                PubSub.publish("dm:readersession:video_started", a);
                break;
            case"displayedFollowTip":
                e.addDisplayedFollowTip(a);
                PubSub.publish("dm:readersession:followtip_displayed", a);
                break;
            default:
                e.addDefaultList(b, a)
            }
        },
        getData: function(a) {
            return d(a)
        },
        addDefaultList: function(h, a, c) {
            if (!c) {
                c = h
            }
            var b = d(c) || [];
            b[b.length] = a;
            f(c, b);
            PubSub.publish("dm:readersession:" + h, a)
        },
        addVideoByLogin: function(b, c) {
            if (typeof c.ownerLogin != "undefined") {
                var a = d(b) || {};
                a[c.ownerLogin] = a[c.ownerLogin] || [];
                if (a[c.ownerLogin].indexOf(c.xid)==-1) {
                    a[c.ownerLogin][a[c.ownerLogin].length] = c.xid;
                    f(b, a)
                }
            }
        },
        addDisplayedFollowTip: function(b) {
            var a = d("displayedFollowTips") || {};
            a[b] = a[b] || 0;
            a[b] += 1;
            f("displayedFollowTips", a)
        },
        __initialize: function() {
            if (!storeWithExpiration.get(e.storeKey)) {
                storeWithExpiration.set(e.storeKey, {}, 86400000)
            }
        }
    };
    return e
}, true);
_DM.define("DM_Alert", function() {
    var b = {
        resendConfirmationRegistration: function(d) {
            var a = d.data("login");
            ajax_call("alert", "resendConfirmationRegistration", a, function() {
                d.parent().fadeOut(function() {
                    d.parent().replaceWith("<p>" + b.resend_email_confirmation + "</p>").fadeIn()
                })
            })
        },
        onResendAlertClicked: function() {
            var d = $(this), a = d.data("type");
            if (a == "registration-confirmation") {
                b.resendConfirmationRegistration(d)
            }
        },
        __initialize: function() {
            $(document).on("click", ".js-resend-mail", b.onResendAlertClicked)
        }
    };
    return b
}, true);
_DM.define("DM_TourLoader", function() {
    var c = {
        welcome: function() {
            return DM_Context.is_logged && d.gkWelcomeTour&&!d.readerEndedWelcomeTour
        }
    }, d = {
        shouldTourTrigerred: function(a) {
            return c[a].apply(this, [a])
        },
        loadWelcomeTour: function() {
            if (getCookie(DM_LoginCookie)) {
                DM_JS.get("smartalert", function() {
                    DM_JS.load("/js/views/shared/tour/welcome.js", "Sd_Tour_Welcome")
                })
            } else {
                PubSub.subscribe("dm:user_cookie:created", function() {
                    DM_JS.get("smartalert", function() {
                        DM_JS.load("/js/views/shared/tour/welcome.js", "Sd_Tour_Welcome")
                    })
                })
            }
        },
        initTours: function() {
            if (d.shouldTourTrigerred("welcome")) {
                DM_JS.load("/js/lib/dm/tour.js", "DM_Tour", function() {
                    d.loadWelcomeTour()
                })
            }
        },
        __onDomReady: function() {
            d.initTours()
        }
    };
    return d
}, true);
_DM.define("Sd_Video_Preview", function() {
    var b = {
        item: null,
        container: null,
        sprite1: null,
        sprite2: null,
        state: 0,
        step: 0,
        total: 0,
        timer: - 1,
        infosTimer: null,
        initEvents: function(d) {
            $(".js-tipsy-ballon").tipsy();
            var a;
            if (d) {
                a = $.isString(d) ? $(d + " div.sd_video_preview") : d instanceof jQuery ? d.find("div.sd_video_preview") : $([])
            } else {
                a = $("div.sd_video_preview")
            }
            a.each(function() {
                if (this.observed) {
                    return 
                }
                this.observed = true;
                $(this).on({
                    mouseenter: b.onPreviewEnter,
                    mouseleave: b.onPreviewLeave
                }).on("click", "span.js-wl_chkbx", Sd_Video_Preview.addToWatchLater)
            })
        },
        onPreviewEnter: function() {
            var a = this;
            b.overTO = setTimeout(function() {
                var e = $(a), f = e.find("div.psprite");
                if (f[0]) {
                    b.onSpriteOver(f)
                }
                b.overTO = null
            }, 200)
        },
        onPreviewLeave: function() {
            if (b.overTO) {
                clearTimeout(b.overTO);
                b.overTO = null;
                return 
            }
            var d = $(this);
            var a = d.find("div.psprite");
            if (a[0]) {
                b.onSpriteOut(a)
            }
        },
        onSpriteOver: function(a) {
            b.$psprite = a;
            b.state = 0;
            b.total = a.hasClass("pmode8") ? 8 : 4;
            window.clearTimeout(b.timer);
            b.timer = window.setTimeout(function() {
                b.spriteAnimate()
            }, 0)
        },
        onSpriteOut: function() {
            window.clearTimeout(b.timer);
            if (b.$container) {
                b.$container.remove();
                b.$container = null
            }
        },
        spriteAnimate: function() {
            switch (this.state) {
            case 0:
                var d = 0;
                if (typeof Sd_Video_PreviewTwig_SpriteOffset != "undefined") {
                    d = Sd_Video_PreviewTwig_SpriteOffset
                }
                this.$img = this.$psprite.parent().find("img");
                this.path = this.$psprite.data("spr");
                this.itemWidth = this.$psprite.width();
                this.itemHeight = this.$psprite.height();
                this.spriteStep = this.itemHeight;
                this.spriteHeight = this.spriteStep * this.total;
                this.spriteTopOffset = Math.floor((this.spriteStep - this.itemHeight) / 2);
                this.$container = $('<div id="psprite_container">').css({
                    position: "absolute",
                    overflow: "hidden",
                    top: - (this.spriteTopOffset + d) + "px",
                    left: d + "px",
                    width: this.itemWidth + "px",
                    height: (this.itemHeight + this.spriteTopOffset) + "px",
                    display: "none"
                });
                this.$sprite2 = $('<img src="' + this.path + '" data-spriteid="2">').css({
                    position: "absolute",
                    border: "none",
                    marginTop: 0,
                    width: this.itemWidth + "px",
                    height: this.spriteHeight + "px",
                    display: "none"
                });
                this.$psprite.before(this.$container.append(this.$sprite2));
                this.state++;
                break;
            case 1:
                var a = this.$sprite2[0];
                if (a.complete || (typeof(a.naturalWidth) != "undefined" && a.naturalWidth !== 0)) {
                    this.$sprite1 = $('<img src="' + this.path + '" data-spriteid="1">').css({
                        position: "absolute",
                        border: "none",
                        marginTop: 0,
                        width: this.itemWidth + "px",
                        height: this.spriteHeight + "px",
                        display: "none"
                    });
                    this.$container.prepend(this.$sprite1).css({
                        display: "block"
                    });
                    this.state++;
                    this.step = 0;
                    this.$sprite2.fadeIn(400)
                }
                break;
            case 2:
                this.step++;
                if (this.step >= this.total) {
                    this.step = 0
                }
                this.$sprite1.show().css({
                    opacity: 1
                });
                if (b.step%2) {
                    this.$sprite1.css({
                        top: "-" + (this.step * this.spriteStep) + "px"
                    });
                    this.$sprite2.fadeOut(400)
                } else {
                    this.$sprite2.css({
                        top: "-" + (this.step * this.spriteStep) + "px"
                    }).fadeIn(400)
                }
                break
            }
            this.timer = window.setTimeout(function() {
                b.spriteAnimate()
            }, this.state == 1 ? 50 : 1000)
        },
        setPlayStatus: function(a) {
            $("div.now_playing").remove();
            $('div.sd_video_preview[data-id="' + a + '"]').append('<div class="preview-message now_playing"><span class="center">' + b.nowPlayingLabel + "</span></div>")
        },
        addToWatchLater: function(j, g) {
            if (j) {
                j.stopPropagation();
                j.preventDefault()
            }
            var a = $(this);
            if (g) {
                if (!a.hasClass("remove")) {
                    DM_WatchLater.add(g, true)
                } else {
                    DM_WatchLater.remove(g, true)
                }
            } else {
                var i = a.closest(".preview_link"), h = i[0].id || i.findParamInClass("id");
                if (!a.data("watched")) {
                    DM_WatchLater.add(h, true)
                }
                if (a.data("watched")) {
                    DM_WatchLater.remove(h, false)
                }
            }
        },
        addedToWatchlater: function(h) {
            if (!h ||!h.length) {
                return 
            }
            var g, a;
            if (h.length === 1) {
                g = $('div.sd_video_preview[data-id="' + h[0] + '"]');
                a = g.find("span.js-wl_chkbx");
                $(".tipsy").remove();
                a.fadeOut(500, function() {
                    g.addClass("in_wl");
                    a.addClass("remove").find(".watch_later").addClass("icon-check font-md").removeClass("icon-time font-lg").end().removeAttr("style original-title").attr("title", b.watchlaterAdded).show("fast")
                });
                a.data("watched", true)
            } else {
                for (var i = 0, j = h.length; i < j; i++) {
                    g = $('div.sd_video_preview[data-id="' + h[i] + '"]');
                    a = g.find("span.js-wl_chkbx");
                    if (!a.hasClass("name")) {
                        a.remove()
                    } else {
                        a.addClass("remove")
                    }
                }
            }
        },
        removedFromWatchlater: function(h) {
            if (!h ||!h.length) {
                return 
            }
            var g, a;
            for (var i = 0, j = h.length; i < j; i++) {
                g = $('div.sd_video_preview[data-id="' + h[i] + '"]');
                a = g.find("span.js-wl_chkbx");
                if (!a.hasClass("name")) {
                    g.removeClass("in_wl");
                    $(".tipsy").remove();
                    a.removeClass("icon-check").find(".watch_later").removeClass("icon-check font-md").addClass("icon-time font-lg").end().removeAttr("style original-title").attr("title", b.watchlaterAdd);
                    a.data("watched", false)
                } else {
                    a.addClass("remove")
                }
            }
        },
        __initialize: function() {
            PubSub.subscribe("video:playing", Sd_Video_Preview.setPlayStatus);
            PubSub.subscribe("watchlater:added", Sd_Video_Preview.addedToWatchlater);
            PubSub.subscribe("watchlater:removed", Sd_Video_Preview.removedFromWatchlater);
            PubSub.subscribe("dm:new_videos_loaded", function(a) {
                b.initEvents(a)
            });
            $(function() {
                b.initEvents()
            })
        }
    };
    return b
}, true);
_DM.define("DM_WatchLater", function() {
    var j = "/controller/Shared_Video_Watchlater?method=", g = function(a) {
        if (!$.isArray(a)) {
            a = [a]
        }
        return a
    }, k = function(a) {
        return i.indexOf(a)==-1 && h.videosFromReq.indexOf(a)!=-1
    }, l = function(a) {
        if (k(a.xid)) {
            h.remove(a.xid)
        }
    }, i = [], h = {
        add: function(a) {
            a = g(a);
            DM_Ajax.jsonCall(j + "add", {
                videoIds: a
            }, function(b) {
                PubSub.publish("watchlater:added", a);
                DM_Stats.trigger({
                    action: "add_to_watchlater"
                })
            }, function(b, c) {
                log("DM_WatchLater", "add failed", b)
            })
        },
        remove: function(a) {
            a = g(a);
            DM_Ajax.jsonCall(j + "remove", {
                videoIds: a,
                next_url: j + "add"
            }, function(b) {
                i = i.concat(a);
                PubSub.publish("watchlater:removed", a)
            }, function(b, c) {
                log("DM_WatchLater", "remove failed", b)
            })
        },
        __initialize: function() {
            PubSub.subscribe("dm:readersession:video_watched", l)
        }
    };
    return h
}, true);
_DM.define("DM_WatchLaterDone", function() {
    var c = function(a) {
        DM_Controller.call("Page_Videos", {
            method: "getWatchlaterModifiedMessage",
            args: [a],
            callback: function(b) {
                PubSub.publish("page:notify", b, "page")
            }
        })
    }, d = {
        __initialize: function() {
            if (getCookie("wl_ok")) {
                deleteCookie("wl_ok");
                c("added")
            }
            PubSub.subscribe("watchlater:added", function(a) {
                c("added")
            });
            PubSub.subscribe("watchlater:removed", function(a) {
                c("removed")
            })
        }
    };
    return d
});
_DM.define("DM_HomeConversionTracker", function() {
    var b = {
        __initialize: function() {
            $("#content").on("click", "a", function() {
                if ($(this).attr("href").match(/^(https?:\/\/[^\/]*dailymotion\.com|\/)/)&&!$(this).attr("href").match(/\/family_filter/)) {
                    ga("send", "event", "home", "convert")
                }
            });
            $("#content").on("play", "div.playable_content", function() {
                ga("send", "event", "home", "convert")
            });
            $("#content").on("click", "button", function() {
                ga("send", "event", "home", "convert")
            })
        }
    };
    return b
}, true);
_DM.define("DM_Masscast", function() {
    var b = {
        enabled: true,
        errorMsg: "FAILED TO REQUEST OAS",
        outframedPositions: ["x11", "x28", "x29", "Top3", "x70", "Middle1"],
        assistedOutframedPositions: ["x11", "Top3", "Middle1", "x28"],
        locked: {},
        hasCompanionAd: false,
        lockPosition: function(a) {
            log("Masscast", "LOCK Position", a);
            b.locked[a] = true
        },
        getCallForPositions: function(a) {
            var d = masscast_async_call;
            if (a && a.length) {
                d = d.replace(/@.+\?/, "@" + a.join(",") + "?")
            }
            return d.replace(/\/[^\/]+@/, "/" + (new Date()).getTime() + "@")
        },
        callSonobi: function() {
            var h = {};
            if (window.sbi_trinity) {
                var g = window.sbi_trinity;
                for (var j in g) {
                    if (g[j] && g[j].hasOwnProperty("sbi_aid")) {
                        var i = g[j].sbi_aid;
                        var a = j.concat("_", g[j].sbi_apoc, "_", g[j].sbi_size);
                        h[a] = 1
                    }
                }
                return h
            }
            return ""
        },
        callOAS: function(f) {
            var e = b.callSonobi();
            if (e !== "") {
                for (var a in e) {
                    if (masscast_async_call.indexOf(a)==-1) {
                        masscast_async_call += "&" + a + "=" + e[a]
                    }
                }
            }
            log("Masscast", "callOAS - arguments :", f);
            $.get(f ? b.getCallForPositions(f) : masscast_async_call).done(function(c) {
                b.callOASCallback(c, f)
            }).fail(function(h, c, d) {
                log("Masscast", b.errorMsg, c, d);
                PubSub.subscribe("adminpanel:loaded", b.reportError)
            })
        },
        reportError: function() {
            $("#masscast_call").prepend('<div><font color="red">' + b.errorMsg + '. Open JS console for details. <a href="#" style="color: red;text-decoration: underline;" onclick="showlog(\'Masscast\'); return false;">Print logs</a> (in JS console).</font></div>')
        },
        parseOASResponse: function(n, o) {
            var m = {};
            if (o && o.length == 1) {
                m[o[0]] = n
            } else {
                var i = n.split('<!--OAS AD="');
                if ($.trim(i[0]) === "") {
                    i.shift()
                }
                if (i.length == 1) {
                    var a = masscast_async_call.match(/@(.+)\?/);
                    if (a) {
                        m[a[1]] = i[0]
                    }
                } else {
                    for (var l = 0, p = i.length; l < p; l++) {
                        var k = i[l].indexOf('"-->');
                        m[i[l].slice(0, k)] = $.trim(i[l].slice(k + 4))
                    }
                }
            }
            return m
        },
        callOASCallback: function(e, a) {
            log("Masscast", "callOASCallback - OASResponse :", e);
            var f = b.parseOASResponse(e, a);
            log("Masscast", "OASResponse parsed :", f);
            $(function() {
                b.asyncHandlePositionCodes(f)
            })
        },
        asyncHandlePositionCodes: function(a) {
            setTimeout(function() {
                b.handlePositionCodes(a)
            }, 200)
        },
        handlePositionCodes: function(k) {
            var l = function(c, d) {
                log("Masscast", "Position", c, ' is "outframed" and need to be run out of an iframe .');
                b.handleOutframedPositionCode(c, d);
                PubSub.publish("masscast:position_status", c, "handled")
            };
            var j = function(c, d) {
                b.updatePositionViaIframe(c, d);
                PubSub.publish("masscast:position_status", c, "handled")
            };
            for (var n in k) {
                if (!k.hasOwnProperty(n)) {
                    continue
                }
                var a = k[n];
                try {
                    var m = $("#mc_" + n);
                    if (b.locked[n]) {
                        log("Masscast", "Position", n, "is locked, update cancelled.");
                        continue
                    }
                    if (!a || a.match(/\/\/ Default ad/)) {
                        m.length && m.empty().hide();
                        log("Masscast", "Position", n, "not found or empty, hide its container from page.");
                        PubSub.publish("masscast:position_status", n, "dismissed");
                        continue
                    }
                    m.length && m.removeClass("mc_hide");
                    if (b.outframedPositions.indexOf(n)>-1 && a.indexOf("OAS_FORCE_IFRAME")===-1) {
                        b.handlePositionWith(n, a, l);
                        continue
                    } else {
                        if (!m.length) {
                            log("Masscast", "Position", n, " container not found, do nothing.");
                            PubSub.publish("masscast:position_status", n, "not_found");
                            continue
                        }
                    }
                    b.handlePositionWith(n, a, j)
                } catch (e) {
                    log("Masscast", "Exception catched while rendering", n, e.message);
                    PubSub.publish("masscast:position_status", n, "error")
                }
            }
            b.locked = {}
        },
        handlePositionWith: function(f, a, e) {
            setTimeout(function() {
                e(f, a)
            }, 100)
        },
        handleOutframedPositionCode: function(d, a) {
            if ($.isString(a)) {
                documentWriteBackup = document.write;
                document.write = function(c) {
                    log("Masscast", "!!! BEWARE !!!", d, "tried to document.write on an already built page :", c)
                };
                (window.execScript || function(c) {
                    window["eval"].call(window, c)
                })(a);
                document.write = documentWriteBackup
            } else {
                log("Masscast", "cannot run position", d, "code:", a)
            }
        },
        updatePositionViaIframe: function(o, p) {
            log("Masscast", "Update position code", o);
            var n = $("#mc_" + o), l = o;
            if (!n.length) {
                return log("Masscast", "Cannot update position via iframe, container not found for ", o)
            }
            if (n.data("default-height")) {
                n.css({
                    height: n.data("default-height")
                }).empty()
            } else {
                n.css({
                    height: 0
                }).empty()
            }
            var a = $('<iframe class="mc_' + l + '_frame" name="mc_' + o + '" allowtransparency="true" framespacing="0" frameborder="no" scrolling="no"></iframe>');
            n.append(a);
            if (o == "Top") {
                $("#content .dmpi_bigbox").css({
                    marginTop: "0"
                })
            }
            var j = a[0].contentDocument || a[0].contentWindow.document, k = "<script>/*" + o + "*/(" + b.preCodeInjection.toString() + ")(window);<\/script>", m = "<script>(" + b.postCodeInjection.toString() + ")(window);<\/script>";
            if (p.indexOf("OAS_HTML")===-1) {
                p = "<script>" + p + "<\/script>"
            }
            j.open().write('<!DOCTYPE html><html><body><style>body { margin:0; padding: 0; text-align:center;}</style><div id="content">' + k + p + m + "</div></body></html>");
            setTimeout(function() {
                j.close()
            }, $.browser.name == "msie" ? 2000 : 0)
        },
        reloadPositionFrame: function(d) {
            if (!b.enabled) {
                log("Masscast", "Masscast has been DISABLED, no reload allowed for", d.name);
                return 
            }
            log("Masscast", "reloadPositionFrame - frame name", d.name);
            var a = d.name.replace(/mc_([a-zA-Z0-9]+)/, "$1");
            log("Masscast", "Frame position to reload:", a);
            b.callOAS([a])
        },
        preCodeInjection: function(a) {
            a.masscast_async_call = parent.masscast_async_call;
            a.inDapIF = true;
            a.ping = parent.ping;
            a.masscast = {
                initReload: function() {
                    if (a.reloadTime && a.parent.DM_Masscast) {
                        a.parent.log("Masscast", "Frame reload initiated for", a.name, ". Reload in", reloadTime, "secs.");
                        setTimeout(function() {
                            a.parent.DM_Masscast.reloadPositionFrame(a)
                        }, reloadTime * 1000)
                    }
                },
                getElementHeight: function(d) {
                    return parseInt(parent.$(d).css("height"), 10)
                },
                adjustHeight: function(j) {
                    var i = masscast.getElementHeight(document.getElementById("content")), h = masscast.getElementHeight(document.body), k = masscast.getElementHeight(frameElement.parentNode), l = (j || 0) || i;
                    if (j) {
                        document.getElementById("content").style.cssText += "position:absolute;top:0;bottom:0;left:0;right:0;"
                    }
                    frameElement.style.height = l + "px";
                    frameElement.parentNode.style.height = "auto";
                    if (l < 50 && k >= 100 && frameElement.name.indexOf("x70")==-1) {
                        frameElement.style.display = "none"
                    } else {
                        if (frameElement.name.indexOf("x52")!=-1) {
                            frameElement.style.display = "auto"
                        } else {
                            frameElement.style.display = "block";
                            if (l < 50 && h >= 100) {
                                frameElement.style.height = h + "px"
                            }
                        }
                    }
                },
                loadDmJs: function(e) {
                    var f = a.document.createElement("script");
                    f.type = "text/javascript";
                    f.src = a.parent.DM_JS.getRealPath(e);
                    f.async = true;
                    a.document.body.appendChild(f)
                },
                render: function(f, e) {
                    a[f + "Parameters"] = e;
                    masscast.loadDmJs("/js/lib/dm/masscast/iframed/" + f + ".js")
                }
            }
        },
        postCodeInjection: function() {
            masscast.initReload();
            function a() {
                a.times = a.times || 1;
                if (window.disableAutoHeight) {
                    parent.log("Masscast", '"AutoHeight" is disabled for', frameElement.name);
                    return frameElement.parentNode.style.height = "auto"
                }
                masscast.adjustHeight();
                if (++a.times < 5) {
                    setTimeout(a, 500)
                } else {
                    parent.log("Masscast", frameElement.name, " iframe height", frameElement.style.height)
                }
            }
            setTimeout(a, 150)
        },
        disable: function() {
            log("Masscast", "Masscast DISABLE");
            b.enabled = false
        },
        companionAdPresent: function() {
            return ($('div[id^="mc_"]').length > 0 && $("#mc_Middle").children().length > 0)
        },
        init: function() {
            if (window.DM_Context.page_type == "player_page") {
                if (b.canCallOAS()) {
                    b.lockPosition("Middle");
                    b.callOAS()
                }
                DM_PlayerEvents.on("AdStart", "video_player_main", function() {
                    var a = setTimeout(function() {
                        if (b.canCallOAS()) {
                            if (!b.companionAdPresent()) {
                                b.locked = {};
                                b.callOAS(["Middle"])
                            }
                        }
                    }, 1000)
                });
                PubSub.subscribe("player:video_start", function() {
                    clearTimeout(timeOut);
                    if (b.canCallOAS()) {
                        if (window.DM_Context.geoip_country != "us" ||!b.companionAdPresent()) {
                            b.locked = {};
                            b.callOAS(["Middle"])
                        }
                    }
                })
            } else {
                if (b.canCallOAS()) {
                    b.callOAS()
                }
            }
        },
        canCallOAS: function() {
            if (window.DM_Context.video_ads === false || window.videolist_has_ads === false) {
                if (window.DM_Context.video_noads_reason === "video attribute|explicit" && window.DM_Context.video_never_ads === false) {
                    log("Masscast", "window.DM_Context.video_noads_reason is explicit, switching call", window.DM_Context.video_noads_reason);
                    window.masscast_async_call = window.masscast_async_ads2_call
                } else {
                    if (window.DM_Context.video_noads_reason === "video attribute|noadfit" && window.DM_Context.video_never_ads === false) {
                        log("Masscast", "window.DM_Context.video_noads_reason is noadfit, switching call", window.DM_Context.video_noads_reason);
                        window.masscast_async_call = window.masscast_async_ads1_call
                    } else {
                        log("Masscast", "window.DM_Context.video_ads is true, blocking ads", window.DM_Context.video_ads);
                        return false
                    }
                }
            }
            return true
        },
        __initialize: function() {
            if (window.masscast_async_call) {
                log("Masscast", "window.masscast_async_call found", window.masscast_async_call);
                PubSub.subscribe("player:rebuffer", function() {
                    log("Masscast", "Player rebuffer catched");
                    b.disable()
                });
                window.masscast = {
                    render: function(a, d) {
                        log("Masscast", "publish", a, "code for assisted handling");
                        PubSub.publish("masscast:position_" + a, d)
                    }
                };
                b.init()
            } else {
                log("Masscast", "window.masscast_async_call not found.")
            }
        }
    };
    return b
}, true);
_DM.define("DM_Masscast_x28", function() {
    var b = {
        params: null,
        domReadyDate: null,
        onCodeReceived: function(a) {
            var d = ((new Date()).getTime() - (this.domReadyDate || new Date()).getTime()) / 1000;
            if (d > 2) {
                return log("Masscast", "Dismiss x28 because it was received more than 2 seconds after DOM ready (", d, " secs).")
            }
            log("Masscast", "Recieved x28 code before DOM ready + 2 secs (", d, " secs). Build intersticial.");
            this.setParams(a);
            this.build()
        },
        setParams: function(a) {
            this.params = $.isString(a) ? (new Function("return " + a))() : a
        },
        build: function() {
            if (this.params&&!this.built) {
                this.built = true;
                window.DM_swftrans_open = this.DM_swftrans_open;
                window.DM_swftrans_close = this.DM_swftrans_close;
                window.DM_Intersticiel_close = this.DM_Intersticiel_close;
                this.hideSiteWrapper();
                this.addHtml();
                this.afterHtml();
                this.setDimmerStyle();
                this.setClickCommand();
                this.addPixelTrackers();
                this.closeTimeout = window.setTimeout(DM_Intersticiel_close, this.params.closeDelay * 1000 || 30000);
                PubSub.publish("masscast:intersticial_opened")
            }
        },
        addHtml: function() {
            $(document.body).prepend(this.getX28Html())
        },
        getX28Html: function() {
            var i = "", a = "", j = "";
            if (this.params.logoImg) {
                i = "background-image: url(" + this.params.logoImg + ")"
            }
            if (this.params.withTopBanner) {
                a = '<div class="x28_top_banner" style="background-color: ' + this.params.withTopBanner.backgroundColor + ';"><div class="x28_top_banner_inner"><div style="' + i + '" id="x28_dimmer_logo" class="no_clkthg"></div><a id="x28_go_to_dm" title="' + (window.DM_Closex28Label || window.DM_CloseLabel || "Close") + '" href="javascript:void(0);" onclick="DM_Intersticiel_close();" class="button medium_size no_clkthg">' + (window.DM_Closex28Label || window.DM_CloseLabel || "Close") + "</a></div></div>"
            } else {
                j = '<a id="x28_go_to_dm" title="' + (window.DM_Closex28Label || window.DM_CloseLabel || "Close") + '" href="javascript:void(0);" onclick="DM_Intersticiel_close();" class="button medium_size no_clkthg">' + (window.DM_Closex28Label || window.DM_CloseLabel || "Close") + '</a><div id="x28_dimmer_logo" class="no_clkthg" style="' + i + '"></div>'
            }
            var h = '<div id="x28_dimmer"' + (a ? ' class="with_banner"' : "") + ">" + a + '<div id="x28_dimmer_inner">' + j + '<div id="intersticiel_object">';
            if (this.params.type == "video") {
                var g = "&autoMute=1&unmuteOnMouseOver=1";
                if (this.params.automute === 0) {
                    g = ""
                }
                h += '<div id="DM_swftr" style="top: ' + (this.params.playerTopOffset || 0) + "px;left: " + (this.params.playerLeftOffset || 0) + 'px;"><iframe frameborder="0" width="' + (this.params.playerWidth || 620) + '" height="' + (this.params.playerHeight || 272) + '" src="http://www.dailymotion.com/embed/video/' + this.params.videoId + "?theme=none&auditude=0&chromeless=1&autoPlay=1&reporting=" + (this.params.reporting || 0) + g + "&related=0&clickThroughURL=" + encodeURIComponent(this.params.clickCommand) + '&clickThroughURLTarget=_blank"></iframe></div>'
            } else {
                if (this.params.type == "flash") {
                    if (window.DM_Flash && window.DM_Flash.VERSION && DM_Flash.VERSION.major !== 0) {
                        h += '<div id="DM_swftr" style="top: ' + this.params.flashTopOffset + 'px;"></div>'
                    } else {
                        h += '<a id="x28_fallback" href="' + this.params.fallbackUrl + '" target="_blank"><img src="' + this.params.fallbackUrl + '" border="0" alt=""></a>'
                    }
                } else {
                    if (this.params.type == "gif") {}
                }
            }
            h += "</div></div></div>";
            return h
        },
        setClickCommand: function() {
            $("#x28_dimmer").on("click", function(a) {
                if (!$(a.target).hasClass("no_clkthg")) {
                    window.open(b.params.clickCommand)
                }
            })
        },
        setDimmerStyle: function() {
            var a = {};
            if (this.params.backgroundUrl) {
                a["background-image"] = "url(" + this.params.backgroundUrl + ")"
            }
            if (this.params.backgroundColor) {
                a["background-color"] = this.params.backgroundColor
            }
            if (this.params.backgroundPosition) {
                a["background-position"] = this.params.backgroundPosition
            }
            $("#x28_dimmer").css(a)
        },
        hideSiteWrapper: function() {
            $("#topwrapper").hide();
            $("#footer").hide()
        },
        afterHtml: function() {
            if (this.params.type == "flash" && window.DM_Flash && window.DM_Flash.VERSION && DM_Flash.VERSION.major !== 0) {
                DM_Flash.build(this.params.flashUrl, "flashtrans", "DM_swftr", this.params.flashWidth, this.params.flashHeight, "#000000", {
                    parameters: {
                        quality: "high",
                        wmode: "transparent",
                        allowScriptAccess: "always"
                    },
                    flashvars: {
                        clickTag: this.params.clickCommand
                    }
                })
            }
        },
        addPixelTrackers: function() {
            if (this.params.pixelsTracking && this.params.pixelsTracking.length) {
                for (var d = 0, a = this.params.pixelsTracking.length; d < a; d++) {
                    ping(this.params.pixelsTracking[d], "x28")
                }
            }
        },
        DM_Intersticiel_close: function() {
            var a = $("#x28_dimmer");
            $("body").off("click");
            if (a) {
                a.off("click").remove();
                $("#topwrapper").show();
                $("#footer").show()
            }
            clearTimeout(b.closeTimeout);
            PubSub.publish("masscast:intersticial_closed")
        },
        DM_swftrans_close: function() {
            $("#DM_swftr").hide();
            $("#DM_backup").show()
        },
        DM_swftrans_open: function() {},
        setDomReadyDate: function() {
            this.domReadyDate = new Date()
        },
        __initialize: function() {
            PubSub.subscribe("masscast:position_x28", this.onCodeReceived, this);
            PubSub.subscribe("dom:ready", this.setDomReadyDate, this)
        }
    };
    return b
}, true);
_DM.define("DM_Masscast_x29", function() {
    var b = {
        params: null,
        domReadyDate: null,
        onCodeReceived: function(a) {
            var d = ((new Date()).getTime() - (this.domReadyDate || new Date()).getTime()) / 1000;
            if (d > 2) {
                return log("Masscast", "Dismiss x29 because it was received more than 2 seconds after DOM ready (", d, " secs).")
            }
            log("Masscast", "Recieved x29 code before DOM ready + 2 secs (", d, " secs). Build intersticial.");
            this.setParams(a);
            this.build()
        },
        setParams: function(a) {
            this.params = $.isString(a) ? (new Function("return " + a))() : a
        },
        build: function() {
            if (this.params&&!this.built) {
                this.built = true;
                window.DM_swftrans_open = this.DM_swftrans_open;
                window.DM_swftrans_close = this.DM_swftrans_close;
                window.DM_Intersticiel_close = this.DM_Intersticiel_close;
                this.hideSiteWrapper();
                this.addHtml();
                this.afterHtml();
                this.setDimmerStyle();
                this.setClickCommand();
                this.addPixelTrackers();
                this.closeTimeout = window.setTimeout(DM_Intersticiel_close, this.params.closeDelay * 1000 || 30000);
                PubSub.publish("masscast:intersticial_opened")
            }
        },
        addHtml: function() {
            $(document.body).prepend(this.getX29Html())
        },
        getX29Html: function() {
            var i = "", a = "", j = "";
            if (this.params.logoImg) {
                i = "background-image: url(" + this.params.logoImg + ")"
            }
            if (this.params.withTopBanner) {
                a = '<div class="x29_top_banner" style="background-color: ' + this.params.withTopBanner.backgroundColor + ';"><div class="x29_top_banner_inner"><div style="' + i + '" id="x29_dimmer_logo" class="no_clkthg"></div><a id="x29_go_to_dm" title="' + (window.DM_Closex29Label || window.DM_CloseLabel || "Close") + '" href="javascript:void(0);" onclick="DM_Intersticiel_close();" class="button medium_size no_clkthg">' + (window.DM_Closex29Label || window.DM_CloseLabel || "Close") + "</a></div></div>"
            } else {
                j = '<a id="x29_go_to_dm" title="' + (window.DM_Closex29Label || window.DM_CloseLabel || "Close") + '" href="javascript:void(0);" onclick="DM_Intersticiel_close();" class="button medium_size no_clkthg">' + (window.DM_Closex29Label || window.DM_CloseLabel || "Close") + '</a><div id="x29_dimmer_logo" class="no_clkthg" style="' + i + '"></div>'
            }
            var h = '<div id="x29_dimmer"' + (a ? ' class="with_banner"' : "") + ">" + a + '<div id="x29_dimmer_inner">' + j + '<div id="intersticiel_object">';
            if (this.params.type == "video") {
                var g = "&autoMute=1&unmuteOnMouseOver=1";
                if (this.params.automute === 0) {
                    g = ""
                }
                h += '<div id="DM_swftr" style="top: ' + (this.params.playerTopOffset || 0) + "px;left: " + (this.params.playerLeftOffset || 0) + 'px;"><iframe frameborder="0" width="' + (this.params.playerWidth || 620) + '" height="' + (this.params.playerHeight || 272) + '" src="http://www.dailymotion.com/embed/video/' + this.params.videoId + "?theme=none&auditude=0&chromeless=1&autoPlay=1&reporting=" + (this.params.reporting || 0) + g + "&related=0&clickThroughURL=" + encodeURIComponent(this.params.clickCommand) + '&clickThroughURLTarget=_blank"></iframe></div>'
            } else {
                if (this.params.type == "flash") {
                    if (window.deconcept && deconcept.SWFObjectUtil.getPlayerVersion().major !== 0) {
                        h += '<div id="DM_swftr" style="top: ' + this.params.flashTopOffset + 'px;"></div>'
                    } else {
                        h += '<a id="x29_fallback" href="' + this.params.fallbackUrl + '" target="_blank"><img src="' + this.params.fallbackUrl + '" border="0" alt=""></a>'
                    }
                } else {
                    if (this.params.type == "gif") {}
                }
            }
            h += "</div></div></div>";
            return h
        },
        setClickCommand: function() {
            $("#x29_dimmer").on("click", function(a) {
                if (!$(a.target).hasClass("no_clkthg")) {
                    window.open(b.params.clickCommand)
                }
            })
        },
        setDimmerStyle: function() {
            var a = {};
            if (this.params.backgroundUrl) {
                a["background-image"] = "url(" + this.params.backgroundUrl + ")"
            }
            if (this.params.backgroundColor) {
                a["background-color"] = this.params.backgroundColor
            }
            if (this.params.backgroundPosition) {
                a["background-position"] = this.params.backgroundPosition
            }
            $("#x29_dimmer").css(a)
        },
        hideSiteWrapper: function() {
            $("#topwrapper").hide();
            $("#footer").hide()
        },
        afterHtml: function() {
            if (this.params.type == "flash" && window.deconcept && deconcept.SWFObjectUtil.getPlayerVersion().major !== 0) {
                myswf = new SWFObject(this.params.flashUrl, "flashtrans", this.params.flashWidth, this.params.flashHeight, 8, "#000000");
                myswf.skipDetect = true;
                myswf.addParam("quality", "high");
                myswf.addParam("wmode", "transparent");
                myswf.addParam("allowscriptaccess", "always");
                myswf.addVariable("clicktag", this.params.clickCommand);
                myswf.write("DM_swftr")
            }
        },
        addPixelTrackers: function() {
            if (this.params.pixelsTracking && this.params.pixelsTracking.length) {
                for (var d = 0, a = this.params.pixelsTracking.length; d < a; d++) {
                    ping(this.params.pixelsTracking[d], "x29")
                }
            }
        },
        DM_Intersticiel_close: function() {
            var a = $("#x29_dimmer");
            $("body").off("click");
            if (a) {
                a.off("click").remove();
                $("#topwrapper").show();
                $("#footer").show()
            }
            clearTimeout(b.closeTimeout);
            PubSub.publish("masscast:intersticial_closed")
        },
        DM_swftrans_close: function() {
            $("#DM_swftr").hide();
            $("#DM_backup").show()
        },
        DM_swftrans_open: function() {},
        setDomReadyDate: function() {
            this.domReadyDate = new Date()
        },
        __initialize: function() {
            PubSub.subscribe("masscast:position_x29", this.onCodeReceived, this);
            PubSub.subscribe("dom:ready", this.setDomReadyDate, this)
        }
    };
    return b
}, true);
_DM.define("DM_Masscast_Top3", function() {
    var b = {
        params: null,
        translateV1ParamstoV2: function(g) {
            var i = {
                skinName: "name",
                trackingTag: "tracking_tag",
                bodyClickUrl: "redirect_url",
                topBannerLink: "redirect_url",
                bodyClickUrlThroughTag: "click_count_urls",
                topBannerLinkThroughTag: "click_count_urls",
                countingPixel: "display_count_urls",
                countingPixelAlt: "display_count_urls"
            };
            var j = {};
            for (var a in i) {
                if (g[a] !== "") {
                    var h = i[a];
                    if (a == "bodyClickUrl") {
                        j.body_redirect_enabled = true
                    }
                    if (h.match(/urls/)) {
                        if (!j[h]) {
                            j[h] = []
                        }
                        if ($.isArray(g[a])) {
                            j[h] = j[h].concat(g[a])
                        } else {
                            j[h].push(g[a])
                        }
                    } else {
                        j[h] = g[a]
                    }
                }
            }
            return j
        },
        onCodeReceived: function(a) {
            this.setParams(a);
            this.build()
        },
        setParams: function(a) {
            this.params = $.isString(a) ? (new Function("return " + a))() : a
        },
        build: function() {
            log("Top3", "Build", this.params);
            try {
                if (typeof this.params == "object") {
                    var a = this;
                    this.params = this.translateV1ParamstoV2(this.params);
                    this.params.from_oas = true;
                    PubSub.publish("masscast:loading_top3");
                    $.getJSON("/controller/Masscast?method=getSkin&method%3Aargs=" + encodeURIComponent($.toJSONString([this.params.name])), function(c) {
                        PubSub.publish("masscast:loaded_top3", c, a.params)
                    })
                }
            } catch (d) {
                log("Top3", "Error while loading skin", d.message, "- skinData:", this.params)
            }
        },
        __initialize: function() {
            PubSub.subscribe("masscast:position_Top3", this.onCodeReceived, this)
        }
    };
    return b
}, true);
_DM.define("DM_Prefetch", function() {
    var b = {
        __initialize: function() {
            if (!window.PREFETCH_CONF) {
                return 
            }
            $(function() {
                setTimeout(b.prefetch, 2000)
            })
        },
        prefetch: function(r) {
            var n, p, q, o = document.createElement("iframe"), a = /\.js\.v[a-f0-9]+$/, m = /\.css\.v[a-f0-9]+$/, i = "";
            o.style.display = "none";
            document.body.appendChild(o);
            for (n = 0, p = PREFETCH_CONF.resources.length; n < p; n++) {
                q = PREFETCH_CONF.resources[n];
                if (q.match(m)) {
                    i += '<link rel="stylesheet" href="' + q + '" />'
                } else {
                    if (q.match(a)) {
                        i += "<script type=\"text/javascript\">(function(d, s){var js, head = d.getElementsByTagName('head')[0];js = d.createElement(s);js.src ='" + q + "';head.appendChild(js);}(document, 'script'));<\/script>"
                    }
                }
            }
            var l = o.contentDocument || o.contentWindow.document;
            l.open().write('<!DOCTYPE html><html><head><script type="text/javascript">__prefetching=true;window.onerror = function(){return true;};<\/script>' + i + "</head><body></body></html>");
            l.close()
        }
    };
    return b
}, true);
_DM.define("Mo_Alert", function() {
    var b = {
        initEvents: function() {
            $(".mo_alert").each(function() {
                var f = $(this);
                var e = f.data("close");
                var a = store.get("alert_close") || {};
                if (e&&!a[e]) {
                    f.removeClass("hidden")
                }
            });
            $("#bodyall").on("click", ".mo_alert .js-alert-close", Mo_Alert.onClose)
        },
        onClose: function() {
            var f = $(this);
            var e = f.closest(".mo_alert");
            e.slideUp();
            if (e.data("close")) {
                var a = store.get("alert_close") || {};
                a[e.data("close")] = true;
                if (e.data("use-cookie")) {
                    setCookie(e.data("close"), true)
                }
                store.set("alert_close", a)
            }
        },
        __initialize: function() {
            b.initEvents()
        }
    };
    return b
}, true);
_DM.define("Sd_User_Tip_Channel", function() {
    var m = [], h = null, l = false, k = false, j = 300, n = 380, i = {
        getGravity: function(a) {
            if (["n", "e", "s", "w"].indexOf(a.data("channel-tip-gravity"))) {
                return a.data("channel-tip-gravity")
            }
            var c = a.offset();
            var b = (c.top - $(window).scrollTop() + j > $(window).height()) ? "s": "n";
            b += (c.left - $(window).scrollLeft() + n > $(window).width()) ? "e" : "w";
            return b
        },
        hideTip: function() {
            if ($(this).data("tipsy")) {
                return 
            }
            k = false
        },
        showTip: function() {
            var a = $(this), b = a.data("user-uri");
            if (m[b] && a.data("tipsy-created")) {
                return 
            }
            k = true;
            if (m[b]) {
                var d = $.parseJSON(m[b]);
                i.createTip(a, d)
            } else {
                var c = {
                    request: a.data("user-uri"),
                    vszone: a.data("tip-vszone"),
                    type: "channel",
                    callback: function(e) {
                        m[b] = JSON.stringify(e);
                        i.createTip(a, e)
                    }
                };
                DM_Controller.call("Shared_User_Tip", c)
            }
        },
        createTip: function(a, c) {
            var b = a.tipsy({
                gravity: i.getGravity(a),
                tooltip_type: "tipsy-infotip",
                className: "tipsy-no-arrow",
                html: true,
                opacity: 1,
                title: function() {
                    return c
                },
                interactive: true,
                live: true,
                delayIn: 500,
                delayOut: 100
            });
            if (k) {
                a.data("tipsy-created", true).trigger("mouseenter")
            }
        },
        onSubscribed: function(a) {
            $elements = $('.js-channel-tip[data-user-uri="/' + a + '"]');
            m["/" + a] = null;
            $elements.map(function() {
                if ($(this).data("tipsy")) {
                    $(this).data("tipsy-updated", true)
                }
            })
        },
        onTipHidden: function(a) {
            $elements = $('.js-channel-tip[data-user-uri="' + a.$element.data("user-uri") + '"]');
            $elements.map(function() {
                if ($(this).data("tipsy-updated")) {
                    $(this).untipsy();
                    $(this).data("tipsy-updated", false);
                    $(this).data("tipsy-created", false)
                }
            })
        },
        onScroll: function() {
            clearTimeout(h);
            var a = function() {
                if (!l) {
                    $elements = $(".js-channel-tip");
                    $elements.map(function() {
                        if ($(this).data("tipsy-created")) {
                            $(this).tipsy("hide");
                            $(this).untipsy();
                            $(this).data("tipsy-updated", false);
                            $(this).data("tipsy-created", false)
                        }
                    })
                }
            };
            h = setTimeout(a, 150)
        },
        onMouseOver: function() {
            l = true
        },
        onMouseOut: function() {
            l = false
        },
        onNewVideosLoaded: function(a) {
            if (a) {
                i.initHandleEvents(a)
            }
        },
        onAdvertisedRelatedLoaded: function() {
            var a = function() {
                i.initHandleEvents($(".pl_video_advertitemrelated, .sd_video_itemrelated"))
            };
            setTimeout(a, 200)
        },
        initHandleEvents: function(a) {
            a.find(".js-channel-tip").on({
                mouseenter: i.showTip,
                mouseleave: i.hideTip,
                mouseover: i.onMouseOver,
                mouseout: i.onMouseOut,
                click: i.onHandleClick,
                scroll: i.onScroll
            })
        },
        initEvents: function() {
            i.initHandleEvents($(document));
            $(document).on("scroll", i.onScroll)
        },
        __onDomReady: function() {
            i.initEvents()
        },
        __initialize: function() {
            PubSub.subscribe("dm:new_videos_loaded", i.onNewVideosLoaded).subscribe("masscast:position_Middle1", i.onAdvertisedRelatedLoaded).subscribe("masscast:position_x11", i.onAdvertisedRelatedLoaded).subscribe("related:loaded", i.onAdvertisedRelatedLoaded).subscribe("dm:user_logged:subscribe", i.onSubscribed).subscribe("dm:user_logged:unsubscribe", i.onSubscribed).subscribe("dm:tipsy:hidden", i.onTipHidden).subscribe("dm:nav:listdisplayed", i.initHandleEvents)
        }
    };
    return i
});
!function(c) {
    var d = function(a, b) {
        this.$element = c(a);
        this.options = b;
        this.list = this.$element.find("ul");
        this.position = 0;
        this.maxPos;
        this.init()
    };
    d.prototype = {
        init: function() {
            var b = this.list.find("li"), a = b.css("margin-right");
            a = typeof a !== "undefined" ? parseInt(a.substr(0, a.length - 2), 10) : 0;
            this.maxPos = b.length - this.options.nbrItems + 1;
            this.itemWidth = b.width() + a;
            this.list.width(this.itemWidth * b.width);
            var f = this;
            this.$element.on("click", "button", function() {
                f.nav.call(f, this)
            });
            if (b.length < this.options.nbrItems) {
                this.$element.find("button").hide()
            }
            this.options.loop || this.$element.find(".prev").hide()
        },
        nav: function(b) {
            var a = c(b).hasClass("prev") ? "prev": "next";
            this[a].call(this);
            typeof this.options.callback === "function" && this.options.callback(this)
        },
        prev: function() {
            var a;
            if (this.position > 0) {
                a = this.itemWidth;
                if (this.options.full) {
                    if (this.position < this.options.nbrItems) {
                        a = this.itemWidth * this.position;
                        this.position = 0
                    } else {
                        this.position -= this.options.nbrItems;
                        a = this.itemWidth * this.options.nbrItems
                    }
                } else {
                    this.position--
                }
                this.list.animate({
                    left: "+=" + a + "px"
                }, this.options.speed);
                this.options.loop || (this.position === 0 && this.$element.find("button.prev").fadeOut(200));
                this.maxPos > this.position - 1 && this.$element.find("button.next").fadeIn()
            } else {
                if (this.options.loop) {
                    a = this.itemWidth * this.maxPos;
                    this.list.animate({
                        left: "-" + a + "px"
                    }, this.options.speed * 2);
                    this.position = this.maxPos
                }
            }
        },
        next: function() {
            if (this.position < this.maxPos) {
                var a = this.itemWidth;
                if (this.options.full) {
                    if (this.position + this.options.nbrItems >= this.maxPos) {
                        a = (this.maxPos - this.position) * this.itemWidth;
                        this.position = this.maxPos
                    } else {
                        a = this.itemWidth * this.options.nbrItems;
                        this.position += this.options.nbrItems
                    }
                } else {
                    this.position++
                }
                this.list.animate({
                    left: "-=" + a + "px"
                }, this.options.speed);
                this.position && this.$element.find(".prev").fadeIn();
                this.options.loop || (this.position === this.maxPos && this.$element.find("button.next").fadeOut(200))
            } else {
                if (this.options.loop) {
                    this.list.animate({
                        left: 0
                    }, this.options.speed * 2);
                    this.position = 0
                }
            }
        }
    };
    c.fn.homepageCarousel = function(a) {
        return this.each(function() {
            a = c.fn.extend(c.fn.homepageCarousel.defaults, typeof a == "object" && a);
            new d(this, a)
        })
    };
    c.fn.homepageCarousel.defaults = {
        nbrItems: 5,
        full: true,
        speed: 400,
        loop: false,
        callback: false
    };
    c.fn.homepageCarousel.Constructor = d
}(window.jQuery);
_DM.define("Pg_WhatToWatch", function() {
    var b = {
        lastScroll: 0,
        pageToLoad: 2,
        categoriesLoaded: false,
        currentChannel: false,
        first_call: false,
        loadMore: function() {
            if (b.firstLoadMoreCall()) {
                return 
            }
            var f = $(this), e = f.find("span"), a = f.find("i");
            if (f.data("loading")) {
                return 
            }
            a.show();
            e.hide();
            f.attr("data-loading", 1);
            DM_Controller.call("Page_WhatToWatch", {
                method: "getFeedAjax",
                args: [b.pageToLoad],
                callback: function(d) {
                    d = JSON.parse(d);
                    if (!!d.html === false || d.hide_spinner) {
                        $("#wtw-load-more").hide()
                    }
                    var c = "wtw-" + ( + new Date()), h = '<div id="' + c + '" style="display: none">' + d.html + "</div>";
                    $(".load-more-button-wrapper").before(h);
                    $("#" + c).fadeIn();
                    f.removeAttr("data-loading");
                    a.hide();
                    e.show();
                    $(".js-tipsy").tipsy();
                    PubSub.publish("dm:new_videos_loaded", $(".pl_wtw_page"));
                    b.pageToLoad++
                }
            })
        },
        loadFeed: function() {
            if (b.firstLoadMoreCall() && $(".preloaded").length) {
                b.showFooter();
                return 
            } else {
                if ($(".preloaded").length === 0) {
                    b.showFooter()
                }
            }
            b.pageIsLoading = true;
            var a = {
                next_page: {
                    append: ".pl_wtw_page",
                    options: {
                        page: b.pageToLoad
                    }
                }
            };
            if (typeof DM_Context != "undefined" && DM_Context.page_channel != "undefined" && DM_Context.page_channel !== "") {
                a.channel = DM_Context.page_channel
            }
            $(".pl_wtw_page").append('<div class="wtw-loader" style="text-align: center"><span class="icon-loading" style="float:none"></span></div>');
            setTimeout(function() {
                DM_WidgetV3.multiGet("whattowatch", document.location.pathname, a, function(d) {
                    $(".pl_wtw_page .wtw-loader").remove();
                    $(".js-tipsy").tipsy();
                    PubSub.publish("dm:new_videos_loaded", $(".pl_wtw_page"));
                    if (JSON.parse(d.responseText).next_page === "") {
                        $(window).off("scroll", b.checkScroll)
                    } else {
                        b.pageToLoad++;
                        b.pageIsLoading = false
                    }
                })
            }, 700)
        },
        firstLoadMoreCall: function() {
            if (b.first_call) {
                b.first_call = false;
                $(".preloaded").show();
                b.pageIsLoading = false;
                return true
            }
            return false
        },
        switchVideosList: function(e) {
            e.preventDefault();
            var f = $(this), a = f.closest(".categories-carousel-item");
            if (f.hasClass("link")) {
                return 
            }
            a.find("header .mo_tabs li").toggleClass("active").find("a").toggleClass("link").toggleClass("alt-link");
            a.find(".carousel-toggle-item").fadeOut(100);
            a.find(".carousel-toggle-item[data-list=" + f.data("list") + "]").fadeIn()
        },
        loadCategories: function() {
            b.categoriesLoaded = true;
            DM_Controller.call("Page_WhatToWatch", {
                method: "getCategoriesCarousel",
                callback: function(a) {
                    a = JSON.parse(a);
                    $("#wtw-categories").html(a.html).fadeIn(300);
                    $(".homepage-carousel").homepageCarousel();
                    PubSub.publish("dm:new_videos_loaded", $("#wtw-categories")).publish("dm:navigation:content:added")
                }
            })
        },
        blacklist: function(e) {
            e && e.preventDefault();
            var h = $(this), j = h.data("action"), i = h.data("target"), a = h.data("video") ? h.data("video"): i;
            if (h.hasClass("wtw-tipsy-link")) {
                h.find("a").click();
                return 
            }
            DM_Controller.call("Page_WhatToWatch", {
                method: "blacklist",
                args: [j, i],
                callback: function(c) {
                    if (c) {
                        var d = $(".pl_wtw_video[data-video=" + a + "]");
                        d.children().fadeOut(300);
                        d.slideUp(500, d.remove)
                    } else {
                        DM_JS.get("DM_Notify", function() {
                            DM_Notify.notify(b.blacklistError, "error")
                        })
                    }
                }
            })
        },
        showBlacklistButton: function() {
            var a = $(".wtw-item-menu-trigger[data-video=" + $(this).data("video") + "]");
            a.addClass("active keep-active")
        },
        hideBlacklistButton: function() {
            var a = $(".wtw-item-menu-trigger[data-video=" + $(this).data("video") + "]");
            a.removeClass("keep-active");
            setTimeout(function() {
                a.hasClass("keep-active") || a.removeClass("active")
            }, 300)
        },
        showBlackListCallback: function(a) {
            $(a.$element.context).addClass("open")
        },
        hideBlacklistCallback: function(a) {
            $(a.$element.context).removeClass("open")
        },
        adjustSizeRecommendedUsersHeight: function() {
            var d = $(".sd_user_recommendedusers");
            if (d.find(".media").length > 5) {
                d.data("height", d.height());
                var a = d.find(".media:first").height() + 20;
                d.height(a * 5)
            }
        },
        loadMoreSuggestions: function(d) {
            var a = $(".sd_user_recommendedusers");
            a.height(a.data("height"));
            $("#wtw-suggestions-load-more").removeClass("loading load-more-button").addClass("loaded more-link");
            $("body").off("click.recommended");
            return false
        },
        initEvents: function() {
            $("body").on("click", "#wtw-load-more", b.loadMore).on("click", ".categories-carousel-item header button", b.switchVideosList).on("mouseenter", ".pl_wtw_video, .wtw-item-blacklist-menu", b.showBlacklistButton).on("mouseleave", ".pl_wtw_video", b.hideBlacklistButton).on("click", ".wtw-item-blacklist-menu li", b.blacklist).on("click", ".categories-carousel-item header .carousel-button", b.switchVideosList).on("click.recommended", "#wtw-suggestions-load-more", b.loadMoreSuggestions);
            if ($("#wtw-categories").length) {
                $(window).on("scroll", function() {
                    b.categoriesLoaded || b.loadCategories()
                })
            }
            PubSub.subscribe("dm:user_logged:subscribe", b.loadMoreSuggestions)
        },
        __initialize: function() {
            b.initEvents();
            b.adjustSizeRecommendedUsersHeight();
            if ($(".preloaded").length) {
                b.first_call = true;
                b.pageToLoad = 3
            }
        }
    };
    return b
}, true);
_DM.define("Pl_Wtw_Video", function() {
    var b = {
        onPlVideoOver: function(a) {
            $(this).find(".icon-close").removeClass("hidden")
        },
        onPlVideoOut: function(a) {
            $(this).find(".icon-close").addClass("hidden")
        },
        onPlVideoClosed: function(a) {
            if (a.parents(".section").children(".js-wtw-closable").not(":hidden").length === 0) {
                a.parents(".section").prev(".page-subtitle").slideUp(200)
            }
        },
        onPlVideoClicked: function(h) {
            var f = $(this), a = f.parents(".js-wtw-closable").data("type"), g = f.parents(".js-wtw-closable").find(".sd_video_preview").data("playable");
            DM_Ajax.jsonCall("/controller/Page_WhatToWatch?method=removeVideoFromList", {
                type: a,
                videoId: g
            }, function(c) {
                b.closeVideo(f)
            }, function(c, d) {
                log("Pl_Wtw_Video", 'Server error while removing video item from list: "' + c + '"');
                console.error("Pl_Wtw_Video", 'Server error while removing video item from list: "' + c + '"');
                b.closeVideo(f)
            })
        },
        closeVideo: function(a) {
            a.parents(".js-wtw-closable").slideUp({
                duration: 100,
                complete: function() {
                    b.onPlVideoClosed(a)
                }
            })
        },
        initEvents: function() {
            $(".js-wtw-closable").on("mouseover", b.onPlVideoOver);
            $(".js-wtw-closable").on("mouseout", b.onPlVideoOut);
            $(".js-wtw-closable .icon-close").on("click", b.onPlVideoClicked)
        },
        __onDomReady: function() {
            b.initEvents()
        }
    };
    return b
});
_DM.define("Sd_User_RecommendedUsers", function() {
    var b = {
        blacklistUser: function() {
            var a = $(this);
            DM_Controller.call("Page_WhatToWatch", {
                method: "blacklist",
                args: ["channel", a.data("target")],
                callback: function(e) {
                    if (e) {
                        var f = a.closest(".user");
                        f.children().fadeOut(300);
                        f.slideUp(500, f.remove)
                    } else {
                        DM_JS.get("DM_Notify", function() {
                            DM_Notify.notify(b.blacklistError, "error")
                        })
                    }
                }
            })
        },
        __initialize: function() {
            $("body").on("click", ".recommended-user-blacklist.logged", b.blacklistUser)
        }
    };
    return b
});
_DM.define("Sd_Pixelle", function() {
    var b = {
        listen: function() {
            pxl("bind", "insert", b.insert)
        },
        insert: function() {
            b.stat("display")
        },
        click: function(a) {
            b.stat("click")
        },
        stat: function(a) {
            DM_Stats.trigger({
                action: "pixelle",
                context: a
            })
        },
        __initialize: function() {
            typeof pxl === "function" && b.listen();
            $("body").on("click", ".pixelle-item a", b.click)
        }
    };
    return b
}, false);
