!function(a, b) {
    var c, d = "superslides";
    c = function(c, d) {
        this.options = b.extend({
            play: !1,
            animation_speed: 600,
            animation_easing: "swing",
            animation: "slide",
            inherit_width_from: a,
            inherit_height_from: a,
            pagination: !0,
            hashchange: !1,
            scrollable: !0,
            elements: {
                preserve: ".preserve",
                nav: ".slides-navigation",
                container: ".slides-container",
                pagination: ".slides-pagination"
            }
        }, d);
        var e = this, f = b("<div>", {
            "class": "slides-control"
        }), g = 1;
        this.$el = b(c), this.$container = this.$el.find(this.options.elements.container);
        var h = function() {
            return g = e._findMultiplier(), e.$el.on("click", e.options.elements.nav + " a", function(a) {
                a.preventDefault(), e.stop(), b(this).hasClass("next") ? e.animate("next", function() {
                    e.start()
                }) : e.animate("prev", function() {
                    e.start()
                })
            }), b(document).on("keyup", function(a) {
                37 === a.keyCode && e.animate("prev"), 39 === a.keyCode && e.animate("next")
            }), b(a).on("resize", function() {
                setTimeout(function() {
                    var a = e.$container.children();
                    e.width = e._findWidth(), e.height = e._findHeight(), a.css({
                        width: e.width,
                        left: e.width
                    }), e.css.containers(), e.css.images()
                }, 10)
            }), b(a).on("hashchange", function() {
                var a, b = e._parseHash();
                a = e._upcomingSlide(b&&!isNaN(b) ? b - 1 : b), a >= 0 && a !== e.current && e.animate(a)
            }), e.pagination._events(), e.start(), e
        }, i = {
            containers: function() {
                e.init ? (e.$el.css({
                    height: e.height
                }), e.$control.css({
                    width: e.width * g,
                    left: - e.width
                }), e.$container.css({})) : (b("body").css({
                    margin: 0
                }), e.$el.css({
                    position: "relative",
                    overflow: "hidden",
                    width: "100%",
                    height: e.height
                }), e.$control.css({
                    position: "relative",
                    transform: "translate3d(0)",
                    height: "100%",
                    width: e.width * g,
                    left: - e.width
                }), e.$container.css({
                    display: "none",
                    margin: "0",
                    padding: "0",
                    listStyle: "none",
                    position: "relative",
                    height: "100%"
                })), 1 === e.size() && e.$el.find(e.options.elements.nav).hide()
            },
            images: function() {
                var a = e.$container.find("img").not(e.options.elements.preserve);
                a.removeAttr("width").removeAttr("height").css({
                    "-webkit-backface-visibility": "hidden",
                    "-ms-interpolation-mode": "bicubic",
                    position: "absolute",
                    left: "0",
                    top: "0",
                    "z-index": "-1",
                    "max-width": "none"
                }), a.each(function() {
                    var a = e.image._aspectRatio(this), c = this;
                    if (b.data(this, "processed"))
                        e.image._scale(c, a), e.image._center(c, a);
                    else {
                        var d = new Image;
                        d.onload = function() {
                            e.image._scale(c, a), e.image._center(c, a), b.data(c, "processed", !0)
                        }, d.src = this.src
                    }
                })
            },
            children: function() {
                var a = e.$container.children();
                a.is("img") && (a.each(function() {
                    if (b(this).is("img")) {
                        b(this).wrap("<div>");
                        var a = b(this).attr("id");
                        b(this).removeAttr("id"), b(this).parent().attr("id", a)
                    }
                }), a = e.$container.children()), e.init || a.css({
                    display: "none",
                    left: 2 * e.width
                }), a.css({
                    position: "absolute",
                    overflow: "hidden",
                    height: "100%",
                    width: e.width,
                    top: 0,
                    zIndex: 0
                })
            }
        }, j = {
            slide: function(a, b) {
                var c = e.$container.children(), d = c.eq(a.upcoming_slide);
                d.css({
                    left: a.upcoming_position,
                    display: "block"
                }), e.$control.animate({
                    left: a.offset
                }, e.options.animation_speed, e.options.animation_easing, function() {
                    e.size() > 1 && (e.$control.css({
                        left: - e.width
                    }), c.eq(a.upcoming_slide).css({
                        left: e.width,
                        zIndex: 2
                    }), a.outgoing_slide >= 0 && c.eq(a.outgoing_slide).css({
                        left: e.width,
                        display: "none",
                        zIndex: 0
                    })), b()
                })
            },
            fade: function(a, b) {
                var c = this, d = c.$container.children(), e = d.eq(a.outgoing_slide), f = d.eq(a.upcoming_slide);
                f.css({
                    left: this.width,
                    opacity: 1,
                    display: "block"
                }), a.outgoing_slide >= 0 ? e.animate({
                    opacity: 0
                }, c.options.animation_speed, c.options.animation_easing, function() {
                    c.size() > 1 && (d.eq(a.upcoming_slide).css({
                        zIndex: 2
                    }), a.outgoing_slide >= 0 && d.eq(a.outgoing_slide).css({
                        opacity: 1,
                        display: "none",
                        zIndex: 0
                    })), b()
                }) : (f.css({
                    zIndex: 2
                }), b())
            }
        };
        j = b.extend(j, b.fn.superslides.fx);
        var k = {
            _centerY: function(a) {
                var c = b(a);
                c.css({
                    top: (e.height - c.height()) / 2
                })
            },
            _centerX: function(a) {
                var c = b(a);
                c.css({
                    left: (e.width - c.width()) / 2
                })
            },
            _center: function(a) {
                e.image._centerX(a), e.image._centerY(a)
            },
            _aspectRatio: function(a) {
                if (!a.naturalHeight&&!a.naturalWidth) {
                    var b = new Image;
                    b.src = a.src, a.naturalHeight = b.height, a.naturalWidth = b.width
                }
                return a.naturalHeight / a.naturalWidth
            },
            _scale: function(a, c) {
                c = c || e.image._aspectRatio(a);
                var d = e.height / e.width, f = b(a);
                f.css(d > c ? {
                    height: e.height,
                    width: e.height / c
                } : {
                    height: e.width * c,
                    width: e.width
                })
            }
        }, l = {
            _setCurrent: function(a) {
                if (e.$pagination) {
                    var b = e.$pagination.children();
                    b.removeClass("current"), b.eq(a).addClass("current")
                }
            },
            _addItem: function(a) {
                var c = a + 1, d = c, f = e.$container.children().eq(a), g = f.attr("id");
                g && (d = g);
                var h = b("<a>", {
                    href: "#" + d,
                    text: d
                });
                h.appendTo(e.$pagination)
            },
            _setup: function() {
                if (e.options.pagination && 1 !== e.size()) {
                    var a = b("<nav>", {
                        "class": e.options.elements.pagination.replace(/^\./, "")
                    });
                    e.$pagination = a.appendTo(e.$el);
                    for (var c = 0; c < e.size(); c++)
                        e.pagination._addItem(c)
                }
            },
            _events: function() {
                e.$el.on("click", e.options.elements.pagination + " a", function(a) {
                    a.preventDefault();
                    var b = e._parseHash(this.hash), c = e._upcomingSlide(b - 1);
                    c !== e.current && e.animate(c, function() {
                        e.start()
                    })
                })
            }
        };
        return this.css = i, this.image = k, this.pagination = l, this.fx = j, this.animation = this.fx[this.options.animation], this.$control = this.$container.wrap(f).parent(".slides-control"), e._findPositions(), e.width = e._findWidth(), e.height = e._findHeight(), this.css.children(), this.css.containers(), this.css.images(), this.pagination._setup(), h()
    }, c.prototype = {
        _findWidth: function() {
            return b(this.options.inherit_width_from).width()
        },
        _findHeight: function() {
            return b(this.options.inherit_height_from).height()
        },
        _findMultiplier: function() {
            return 1 === this.size() ? 1 : 3
        },
        _upcomingSlide: function(a) {
            if (/next/.test(a))
                return this._nextInDom();
            if (/prev/.test(a))
                return this._prevInDom();
            if (/\d/.test(a))
                return + a;
            if (a && /\w/.test(a)) {
                var b = this._findSlideById(a);
                return b >= 0 ? b : 0
            }
            return 0
        },
        _findSlideById: function(a) {
            return this.$container.find("#" + a).index()
        },
        _findPositions: function(a, b) {
            b = b || this, void 0 === a && (a =- 1), b.current = a, b.next = b._nextInDom(), b.prev = b._prevInDom()
        },
        _nextInDom: function() {
            var a = this.current + 1;
            return a === this.size() && (a = 0), a
        },
        _prevInDom: function() {
            var a = this.current - 1;
            return 0 > a && (a = this.size() - 1), a
        },
        _parseHash: function(b) {
            return b = b || a.location.hash, b = b.replace(/^#/, ""), b&&!isNaN( + b) && (b =+ b), b
        },
        size: function() {
            return this.$container.children().length
        },
        destroy: function() {
            return this.$el.removeData()
        },
        update: function() {
            this.css.children(), this.css.containers(), this.css.images(), this.pagination._addItem(this.size()), this._findPositions(this.current), this.$el.trigger("updated.slides")
        },
        stop: function() {
            clearInterval(this.play_id), delete this.play_id, this.$el.trigger("stopped.slides")
        },
        start: function() {
            var c = this;
            c.options.hashchange ? b(a).trigger("hashchange") : this.animate(), this.options.play && (this.play_id && this.stop(), this.play_id = setInterval(function() {
                c.animate()
            }, this.options.play)), this.$el.trigger("started.slides")
        },
        animate: function(b, c) {
            var d = this, e = {};
            if (!(this.animating || (this.animating=!0, void 0 === b && (b = "next"), e.upcoming_slide = this._upcomingSlide(b), e.upcoming_slide >= this.size()))
                ) {
                if (e.outgoing_slide = this.current, e.upcoming_position = 2 * this.width, e.offset =- e.upcoming_position, ("prev" === b || b < e.outgoing_slide) && (e.upcoming_position = 0, e.offset = 0), d.size() > 1 && d.pagination._setCurrent(e.upcoming_slide), d.options.hashchange) {
                    var f = e.upcoming_slide + 1, g = d.$container.children(":eq(" + e.upcoming_slide + ")").attr("id");
                    a.location.hash = g ? g : f
                }
                d.$el.trigger("animating.slides", [e]), d.animation(e, function() {
                    d._findPositions(e.upcoming_slide, d), "function" == typeof c && c(), d.animating=!1, d.$el.trigger("animated.slides"), d.init || (d.$el.trigger("init.slides"), d.init=!0, d.$container.fadeIn("fast"))
                })
            }
        }
    }, b.fn[d] = function(a, e) {
        var f = [];
        return this.each(function() {
            var g, h, i;
            return g = b(this), h = g.data(d), i = "object" == typeof a && a, h || (f = g.data(d, h = new c(this, i))), "string" == typeof a && (f = h[a], "function" == typeof f) ? f = f.call(h, e) : void 0
        }), f
    }, b.fn[d].fx = {}
}(this, window[jimdoData.jQuery]);
