(function(e) {
    e.fn.drag = function(t, n, r) {
        var i = typeof t == "string" ? t: "", s = e.isFunction(t) ? t: e.isFunction(n) ? n: null;
        i.indexOf("drag") !== 0 && (i = "drag" + i);
        r = (t == s ? n : r) || {};
        return s ? this.bind(i, r, s) : this.trigger(i)
    };
    var t = e.event, n = t.special, r = n.drag = {
        defaults: {
            which: 1,
            distance: 0,
            not: ":input",
            handle: null,
            relative: !1,
            drop: !0,
            click: !1
        },
        datakey: "dragdata",
        noBubble: !0,
        add: function(t) {
            var n = e.data(this, r.datakey), i = t.data || {};
            n.related += 1;
            e.each(r.defaults, function(e, t) {
                i[e] !== undefined && (n[e] = i[e])
            })
        },
        remove: function() {
            e.data(this, r.datakey).related -= 1
        },
        setup: function() {
            if (e.data(this, r.datakey))
                return;
            var n = e.extend({
                related: 0
            }, r.defaults);
            e.data(this, r.datakey, n);
            t.add(this, "touchstart mousedown", r.init, n);
            this.attachEvent && this.attachEvent("ondragstart", r.dontstart)
        },
        teardown: function() {
            var n = e.data(this, r.datakey) || {};
            if (n.related)
                return;
            e.removeData(this, r.datakey);
            t.remove(this, "touchstart mousedown", r.init);
            r.textselect(!0);
            this.detachEvent && this.detachEvent("ondragstart", r.dontstart)
        },
        init: function(i) {
            if (r.touched)
                return;
            var s = i.data, o;
            if (i.which != 0 && s.which > 0 && i.which != s.which)
                return;
            if (e(i.target).is(s.not))
                return;
            if (s.handle&&!e(i.target).closest(s.handle, i.currentTarget).length)
                return;
            r.touched = i.type == "touchstart" ? this : null;
            s.propagates = 1;
            s.mousedown = this;
            s.interactions = [r.interaction(this, s)];
            s.target = i.target;
            s.pageX = i.pageX;
            s.pageY = i.pageY;
            s.dragging = null;
            o = r.hijack(i, "draginit", s);
            if (!s.propagates)
                return;
            o = r.flatten(o);
            if (o && o.length) {
                s.interactions = [];
                e.each(o, function() {
                    s.interactions.push(r.interaction(this, s))
                })
            }
            s.propagates = s.interactions.length;
            s.drop!==!1 && n.drop && n.drop.handler(i, s);
            r.textselect(!1);
            r.touched ? t.add(r.touched, "touchmove touchend", r.handler, s) : t.add(document, "mousemove mouseup", r.handler, s);
            if (!r.touched || s.live)
                return !1
        },
        interaction: function(t, n) {
            var i = e(t)[n.relative ? "position": "offset"]() || {
                top: 0,
                left: 0
            };
            return {
                drag: t,
                callback: new r.callback,
                droppable: [],
                offset: i
            }
        },
        handler: function(i) {
            var s = i.data;
            switch (i.type) {
            case!s.dragging && "touchmove":
                i.preventDefault();
            case!s.dragging && "mousemove":
                if (Math.pow(i.pageX - s.pageX, 2) + Math.pow(i.pageY - s.pageY, 2) < Math.pow(s.distance, 2))
                    break;
                i.target = s.target;
                r.hijack(i, "dragstart", s);
                s.propagates && (s.dragging=!0);
            case"touchmove":
                i.preventDefault();
            case"mousemove":
                if (s.dragging) {
                    r.hijack(i, "drag", s);
                    if (s.propagates) {
                        s.drop!==!1 && n.drop && n.drop.handler(i, s);
                        break
                    }
                    i.type = "mouseup"
                };
            case"touchend":
            case"mouseup":
            default:
                r.touched ? t.remove(r.touched, "touchmove touchend", r.handler) : t.remove(document, "mousemove mouseup", r.handler);
                if (s.dragging) {
                    s.drop!==!1 && n.drop && n.drop.handler(i, s);
                    r.hijack(i, "dragend", s)
                }
                r.textselect(!0);
                s.click===!1 && s.dragging && e.data(s.mousedown, "suppress.click", (new Date).getTime() + 5);
                s.dragging = r.touched=!1
            }
        },
        hijack: function(n, i, s, o, u) {
            if (!s)
                return;
            var a = {
                event: n.originalEvent,
                type: n.type
            }, f = i.indexOf("drop") ? "drag": "drop", l, c = o || 0, h, p, d, v = isNaN(o) ? s.interactions.length: o;
            n.type = i;
            n.originalEvent = null;
            s.results = [];
            do 
                if (h = s.interactions[c]) {
                    if (i !== "dragend" && h.cancelled)
                        continue;
                        d = r.properties(n, s, h);
                        h.results = [];
                        e(u || h[f] || s.droppable).each(function(o, u) {
                            d.target = u;
                            n.isPropagationStopped = function() {
                                return !1
                            };
                            l = u ? t.dispatch.call(u, n, d) : null;
                            if (l===!1) {
                                if (f == "drag") {
                                    h.cancelled=!0;
                                    s.propagates -= 1
                                }
                                i == "drop" && (h[f][o] = null)
                            } else 
                                i == "dropinit" && h.droppable.push(r.element(l) || u);
                                i == "dragstart" && (h.proxy = e(r.element(l) || h.drag)[0]);
                                h.results.push(l);
                                delete n.result;
                                if (i !== "dropinit")
                                    return l
                                });
                                s.results[c] = r.flatten(h.results);
                                i == "dropinit" && (h.droppable = r.flatten(h.droppable));
                                i == "dragstart"&&!h.cancelled && d.update()
                }
            while (++c < v);
            n.type = a.type;
            n.originalEvent = a.event;
            return r.flatten(s.results)
        },
        properties: function(e, t, n) {
            var i = n.callback;
            i.drag = n.drag;
            i.proxy = n.proxy || n.drag;
            i.startX = t.pageX;
            i.startY = t.pageY;
            i.deltaX = e.pageX - t.pageX;
            i.deltaY = e.pageY - t.pageY;
            i.originalX = n.offset.left;
            i.originalY = n.offset.top;
            i.offsetX = i.originalX + i.deltaX;
            i.offsetY = i.originalY + i.deltaY;
            i.drop = r.flatten((n.drop || []).slice());
            i.available = r.flatten((n.droppable || []).slice());
            return i
        },
        element: function(e) {
            if (e && (e.jquery || e.nodeType == 1))
                return e
        },
        flatten: function(t) {
            return e.map(t, function(t) {
                return t && t.jquery ? e.makeArray(t) : t && t.length ? r.flatten(t) : t
            })
        },
        textselect: function(t) {
            e(document)[t ? "unbind": "bind"]("selectstart", r.dontstart).css("MozUserSelect", t ? "" : "none");
            document.unselectable = t ? "off" : "on"
        },
        dontstart: function() {
            return !1
        },
        callback: function() {}
    };
    r.callback.prototype = {
        update: function() {
            n.drop && this.available.length && e.each(this.available, function(e) {
                n.drop.locate(this, e)
            })
        }
    };
    var i = t.dispatch;
    t.dispatch = function(t) {
        if (t && "type"in t && e.data(this, "suppress." + t.type) - (new Date).getTime() > 0) {
            e.removeData(this, "suppress." + t.type);
            return 
        }
        return i.apply(this, arguments)
    };
    var s = t.fixHooks.touchstart = t.fixHooks.touchmove = t.fixHooks.touchend = t.fixHooks.touchcancel = {
        props: "clientX clientY pageX pageY screenX screenY".split(" "),
        filter: function(t, n) {
            if (n) {
                var r = n.touches && n.touches[0] || n.changedTouches && n.changedTouches[0] || null;
                r && e.each(s.props, function(e, n) {
                    t[n] = r[n]
                })
            }
            return t
        }
    };
    n.draginit = n.dragstart = n.dragend = r
})(jQuery);
$.getObject("Pb.Component.Element.Toggle", !0).Toggle = function() {
    function e(e) {
        this.init(e)
    }
    var t = {
        container: null,
        dragging: !1,
        init: function(e) {
            if (!e.container)
                return;
            this.container = $(e.container);
            this.bindEvents()
        },
        bindEvents: function() {
            var e = $(this.container).find(".ball"), t = $(this.container).find(".toggle");
            if (e.length && t.length) {
                this.initialOffset = e.offset().left;
                this.parentOffset = this.container.offset().left;
                e.drag("start", $.proxy(function(t, n) {
                    this.mouseOffset = t.pageX - e.offset().left;
                    n.limit = {};
                    n.limit.left = 3;
                    n.limit.right = 35;
                    e.addClass("dragging")
                }, this)).drag($.proxy(function(t, n) {
                    var r = t.pageX - this.parentOffset - this.mouseOffset, i = r;
                    i < n.limit.left ? i = n.limit.left : i > n.limit.right && (i = n.limit.right);
                    e.css({
                        left: i
                    })
                }, this)).drag("end", $.proxy(function(n, r) {
                    this.preventClick=!0;
                    n.stopPropagation();
                    e.removeClass("dragging");
                    var i = n.pageX - this.parentOffset, s = t.hasClass("on");
                    i > r.limit.left + 12&&!s || i < r.limit.right - 12 && s ? this.toggle() : s ? e.css("left", r.limit.right) : e.css("left", r.limit.left)
                }, this));
                t.on("click", $.proxy(function() {
                    this.preventClick ? this.preventClick=!1 : this.toggle()
                }, this))
            }
        },
        toggle: function() {
            var e = $(this.container).find(".ball"), t = $(this.container).find(".ballOn"), n = $(this.container).find(".ballOff"), r = $(this.container).find(".toggle");
            if (!r.hasClass("on")) {
                r.addClass("on");
                e.css("left", "35px");
                t.css("display", "block");
                n.hide();
                this.container.trigger("Pb.Component.Element.Toggle.Toggle::on")
            } else {
                r.removeClass("on");
                e.css("left", "3px");
                t.hide();
                n.css("display", "block");
                this.container.trigger("Pb.Component.Element.Toggle.Toggle::off")
            }
        }
    };
    $.extend(e.prototype, t);
    return e
}();
$.getObject("Pb.Component.Print", !0).PrintLink = function() {
    function e(e) {
        this.init(e);
        this.bindAction()
    }
    var t = {
        init: function(e) {
            $(document).trigger($.Event("ActionEventsDispatcher::registerevent", {
                pbEvent: {
                    eventSelector: ".printAction",
                    eventType: "click",
                    event: "PrintLink::Print"
                }
            }));
            if (typeof PIO != "undefined") {
                $("#cartMenu .count").html(PIO.getNumItems());
                PIO.config({
                    recipeId: Pb.Data.Shared.get("pioConfig").isAol ? Pb.Data.Shared.get("pioConfig").aolRecipeId: Pb.Data.Shared.get("pioConfig").recipeId,
                    url: Pb.Data.Shared.get("pioConfig").url,
                    preload: {
                        on: !0
                    }
                })
            }
            $(window).on("popstate", $.proxy(function(e) {
                var t = e.state || e.originalEvent.state;
                typeof PIO != "undefined" && t && t.category&&!t.product && PIO.close()
            }, this));
            $(document).on("click", ".printCurrency", function(e) {
                e.preventDefault();
                PIO.msg("show-currencies")
            }).on("click", ".printUnits", function(e) {
                e.preventDefault();
                PIO.msg("show-measures")
            }).on("click", ".printShipping", function(e) {
                e.preventDefault();
                PIO.msg("change-country")
            })
        },
        bindAction: function() {
            $(document).on("PrintLink::Print", $.proxy(function(e) {
                if ($(e.pbEventTarget).data("name") === "Flat Cards")
                    window.location = "/print/greetingcards";
                else if (typeof PIO != "undefined") {
                    var t = this.buildPrintData(e);
                    if (t) {
                        PIO.open(t);
                        Pb.Track.tr("print_widget_open");
                        $("html, body").animate({
                            scrollTop: 0
                        }, 750)
                    }
                }
            }, this))
        },
        buildPrintData: function(e) {
            var t, n = e.pbData, r = $(e.pbEventTarget), i = {}, s = Pb.Data.Shared.get("loggedInUser"), o = Pb.Data.Shared.get("pioConfig"), u = Pb.Data.get("secureUrl") + o.cssUrl, a = this, f = Pb.Data.Shared.get("pioConfig").url.replace("/widget/", "").replace("http://", "").replace("https://", "");
            if (s && s.baseCobaltUri) {
                s.baseCobaltUri.substr(s.baseCobaltUri.length - 1) !== "/" && (s.baseCobaltUri += "/");
                i = {
                    pb_token: o.access_token,
                    pb_refresh_token: o.refresh_token,
                    pb_username: s.username,
                    pb_base_uri: s.baseCobaltUri
                }
            }
            t = {
                recipeId: Pb.Data.Shared.get("pioConfig").isAol ? Pb.Data.Shared.get("pioConfig").aolRecipeId: Pb.Data.Shared.get("pioConfig").recipeId,
                canUseMap: !1,
                vars: i,
                url: Pb.Data.Shared.get("pioConfig").url,
                helpUrl: "http://support.photobucket.com/hc/en-us/categories/200154330-Print",
                embedInElement: document.getElementById("main"),
                productsUrl: Pb.Data.get("secureUrl") + "print",
                confirmationUrl: Pb.Data.get("secureUrl") + "print/confirmation",
                usePushState: !!history.pushState,
                fns: {
                    onOpen: $.proxy(function() {
                        $("#main > div, .hideInWidget").hide();
                        $(".showInWidget").show();
                        $("#footer, #main").addClass("inWidget");
                        PIO.getNumItems() && Pb.PageTracker.addGoogleAnalyticsPartnerEvent(Pb.Data.Shared.get("pioConfig").googleAccount, f, "print.io", "widget", "opened:With Items", PIO.getNumItems());
                        a.addPrintAbandonmentWarning()
                    }, a),
                    onClose: $.proxy(function(e) {
                        return a.currentTemplate === "tpl-product-a" || a.currentTemplate === "tpl-product-b" || a.orderId || a.currentTemplate === "tpl-cart"?!0 : confirm("Any unsaved changes or items not in your cart will be lost. Are you sure you want to leave?")
                    }, a),
                    onCartChange: $.proxy(function(e) {
                        $.cookie("pio", PIO.getNumItems(), {
                            expires: 7,
                            domain: ".photobucket.com",
                            path: "/"
                        });
                        $(document).trigger("Printio::cartChanged")
                    }, a),
                    onEvent: function(e, t) {
                        if (typeof console != "undefined" && Pb.Data.get("version") === "development") {
                            console.log("key: " + e);
                            console.log(t);
                            console.log("-------------------------------------------------")
                        }
                        if (e === "close") {
                            $("#main > div, .hideInCart, .hideInWidget").show();
                            $(".showInWidget, .showInCart").hide();
                            $("#footer, #main").removeClass("inWidget");
                            PIO.getNumItems() && Pb.PageTracker.addGoogleAnalyticsPartnerEvent(Pb.Data.Shared.get("pioConfig").googleAccount, f, "print.io", "widget", "closed:With Items", PIO.getNumItems());
                            a.removePrintAbandonmentWarning()
                        }
                        if (e === "section-init") {
                            a.currentTemplate = t.tplName;
                            if (t.tplName === "tpl-cart" || t.tplName === "tpl-shipping" || t.tplName === "tpl-shipment-review" || t.tplName === "tpl-address-validation" || t.tplName === "tpl-billing-a") {
                                $(".hideInCart").hide();
                                $(".showInCart").css("display", "block")
                            } else {
                                $(".hideInCart").show();
                                $(".showInCart").hide()
                            }
                        }
                        e === "analytics-pageview" && Pb.PageTracker.addGoogleAnalyticsPartnerPageView(Pb.Data.Shared.get("pioConfig").googleAccount, f, t.page, t.title, "print.io");
                        if (e === "analytics-send") {
                            PIO.close();
                            $("body").html("");
                            if (!a.paypal) {
                                document.location.href = "/print/confirmation?orderid=" + a.orderId;
                                return 
                            }
                        }
                        e === "paypal" && (a.paypal=!0);
                        if (e === "onOrderSubmit") {
                            a.orderId = t.id;
                            $.cookie("pio", null, {
                                expires: - 1,
                                domain: ".photobucket.com",
                                path: "/"
                            })
                        }
                    }
                }
            };
            typeof u != "undefined" && (t.style = {
                cssUrl: u
            });
            if (r) {
                r.data("product") && $.extend(t, {
                    items: [{
                        productId: r.data("product"),
                        templateName: r.data("templateid") ? r.data("templateid"): "",
                        sku: r.data("sku") ? r.data("sku"): ""
                    }
                    ],
                    goTo: "tpl-product-b"
                });
                r.data("customize") && $.extend(t, {
                    goToCustomize: !0
                });
                (r.data("template") || e.template) && $.extend(t, {
                    goTo: e.template ? e.template: r.data("template")
                })
            }
            if (e.printProductTemplateData) {
                var l = e.printProductTemplateData;
                $.extend(t, {
                    items: [{
                        productId: l.product,
                        templateName: l.templateName,
                        sku: l.sku
                    }
                    ],
                    goTo: l.goto
                })
            }
            if (!n)
                return t;
            switch (n.pbdkey) {
            case Pb.Data.Shared.MEDIA:
                var c = Pb.Data.Shared.get(Pb.Data.Shared.MEDIA);
                if (c.mediaType === "image" && c.cp) {
                    $.extend(t, {
                        images: [c.originalUrl.replace("http://i", "http://oi")]
                    });
                    return t
                }
                break;
            case Pb.Data.Shared.ALBUM:
                var h = Pb.Data.Shared.get(Pb.Data.Shared.ALBUM), p = "/api/user/" + h.ownername + "/album/" + h.location + "/media/print";
                Pb.Ajax.ajax({
                    url: p,
                    type: "get",
                    success: function(e) {
                        $.extend(t, {
                            images: e.data
                        });
                        PIO.open(t);
                        Pb.Track.tr("print_widget_open")
                    },
                    error: function() {
                        PIO.open(t);
                        Pb.Track.tr("print_widget_open")
                    },
                    messageHandler: function() {
                        return !0
                    }
                });
                break;
            default:
                return t
            }
        },
        warnOnPageUnload: function(e) {
            if (this.currentTemplate === "tpl-product-a" || this.currentTemplate === "tpl-product-b" || this.orderId || this.currentTemplate === "tpl-cart")
                return;
            return e.returnValue = 'Any unsaved changes or items not in your cart will be lost. Are you sure you want to leave?"'
        },
        addPrintAbandonmentWarning: function() {
            this.pageLevelAbandonmentEvents = [];
            $._data(window, "events").beforeunload && $.each($._data(window, "events").beforeunload, $.proxy(function(e, t) {
                if (t) {
                    this.pageLevelAbandonmentEvents.push(t);
                    t.namespace ? $(window).off("beforeunload." + t.namespace) : $(window).off("beforeunload")
                }
            }, this));
            $(window).on("beforeunload.printAbandonment", $.proxy(this.warnOnPageUnload, this))
        },
        removePrintAbandonmentWarning: function() {
            $(window).off(".printAbandonment");
            this.pageLevelAbandonmentEvents && this.pageLevelAbandonmentEvents.length && $.each(this.pageLevelAbandonmentEvents, function(e, t) {
                t.namespace ? $(window).on("beforeunload." + t.namespace, t.handler) : $(window).on("beforeunload", t.handler)
            })
        }
    }, n;
    $.extend(e.prototype, t);
    Pb.InitEventQueue.addToDomReady(function() {
        n = new e
    }, Pb.InitEventQueue.HIGH_PRIORITY);
    return e
}();
