jQuery(function($) {

    function wpexMainMenuDrops() {
        $('ul.dropdown-menu li').hover(function() {
            $(this).children('ul').stop(true, true).fadeIn(300);
        }, function() {
            $(this).children('ul').stop(true, true).fadeOut(300);
        });
    }

    function wpexFixedHeader() {
        $("#masthead.fixed-scroll").sticky({
            topSpacing: 0,
            className: 'sticky-header'
        });
    }

    function wpexFadeInSection() {
        $('.fade-in-section').waypoint(function(direction) {
            $(this).find('article').each(function(index) {
                $(this).delay(120 * index).animate({
                    "opacity": 1
                }, 1000);
            });
        }, {
            offset: '80%'
        });
        $('.fade-in-home-clients').waypoint(function(direction) {
            $(this).find('article').each(function(index) {
                $(this).delay(140 * index).animate({
                    "opacity": 1
                }, "normal");
            });
        }, {
            offset: '60%'
        });
    }

    function wpexFadeInElements() {
        offset = ($('.fade-in').data('fade_ofset')) ? $('.fade-in').data('fade_ofset') : '50';
        $('.fade-in').waypoint(function(direction) {
            $(this).animate({
                "opacity": 1,
                "top": 0
            }, "slow");
        }, {
            offset: offset + '%'
        });
    }

    function wpexParallaxBg() {
        $('.style-parallax').each(function() {
            var $bgobj = $(this);
            $window = $(window);
            $(window).scroll(function() {
                var yPos = - ($window.scrollTop() / '10');
                var coords = '50% ' + yPos + 'px';
                $bgobj.css({
                    backgroundPosition: coords
                });
            });
        });
    }

    function wpexBackTopScroll() {
        $('.site-scroll-top').on('click', function() {
            $('html, body').animate({
                scrollTop: 0
            }, 400);
            return false;
        });
    }

    function wpexCommentScroll() {
        $('.comment-scroll a').click(function(event) {
            event.preventDefault();
            $('html,body').animate({
                scrollTop: $(this.hash).offset().top - 180
            }, 'normal');
        });
    }

    function wpexPortOverlay() {
        $('.portfolio-entry-overlay-inner').each(function() {
            var overlayHeight = $(this).height();
            $(this).css({
                marginTop: - overlayHeight / 2
            });
        });
    }

    function wpexLightbox() {
        /*lightbox*/
        $('.wpex-lightbox').magnificPopup({
            type: 'image'
        });
        $('.wpex-gallery-lightbox, .images .zoom').magnificPopup({
            type: 'image',
            gallery: {
                enabled: true
            }
        });
    }

    function wpexTestimonialsSlider() {
        $('.testimonials-slider-loader').hide();
        $('.testimonials-slider').flexslider({
            slideshow: false,
            randomize: false,
            animation: 'slide',
            animationLoop: true,
            direction: "horizontal",
            reverse: false,
            initDelay: 0,
            slideshowSpeed: 7000,
            animationSpeed: 500,
            smoothHeight: true,
            controlNav: false,
            pauseOnAction: true,
            pauseOnHover: false,
            useCSS: true,
            touch: false,
            video: false,
            directionNav: false,
            controlNav: "thumbnails",
            controlsContainer: ".testimonials-slider",
        });
    }

    function wpexFaqScroll() {
        if ($('#faq-toc').width()) {
            $('#faq-toc').onePageNav({
                currentClass: 'active',
                changeHash: false,
                scrollSpeed: 400,
                scrollOffset: 210,
                scrollThreshold: 0.5,
                filter: '',
                begin: function() {
                    //I get fired when the animation is starting
                },
                end: function() {
                    //I get fired when the animation is ending
                },
                scrollChange: function($currentListItem) {
                    //I get fired when you enter a section and I pass the list item of the section
                }
            });
        }
    }

    function wpexGalleryFormatSlider() {
        $('.gallery-format-post-slider').flexslider({
            animation: 'fade',
            animationSpeed: 500,
            slideshow: true,
            smoothHeight: true,
            controlNav: false,
            prevText: '<i class="icon-chevron-left"></i>',
            nextText: '<i class="icon-chevron-right"></i>',
            start: function(slider) {
                $('.gallery-format-post-slider li.slide').click(function(event) {
                    event.preventDefault();
                    slider.flexAnimate(slider.getTarget("next"));
                });
            }
        });
    }

    function wpexTipsy() {
        $(function() {
            $('a.wpex-tipsy-s').tipsy({
                fade: true,
                gravity: 's'
            });
        });
    }

    function wpexResponsiveNavTop() {
        $("<select />").appendTo("#top-bar-nav");
        $("<option />", {
            "selected": "selected",
            "value": "",
            "text": wpexLocalize.responsiveMenuText
        }).appendTo("#top-bar-nav select");

        $("#top-bar-nav a").each(function() {
            var el = $(this);
            if (el.parents('.sub-menu').length >= 1) {
                $('<option />', {
                    'value': el.attr("href"),
                    'text': '- ' + el.text()
                }).appendTo("#top-bar-nav select");
            } else if (el.parents('.sub-menu .sub-menu').length >= 1) {
                $('<option />', {
                    'value': el.attr('href'),
                    'text': '-- ' + el.text()
                }).appendTo("#top-bar-nav select");
            } else {
                $('<option />', {
                    'value': el.attr('href'),
                    'text': el.text()
                }).appendTo("#top-bar-nav select");
            }
        });
        $("#top-bar-nav select").change(function() {
            window.location = $(this).find("option:selected").val();
        });

        $("#top-bar-nav select").uniform();
    }

    function wpexResponsiveNav() {
        $("<select />").appendTo("#site-navigation");
        $("<option />", {
            "selected": "selected",
            "value": "",
            "text": wpexLocalize.responsiveMenuText
        }).appendTo("#site-navigation select");

        $("#site-navigation a").each(function() {
            var el = $(this);
            if (el.parents('.sub-menu').length >= 1) {
                $('<option />', {
                    'value': el.attr("href"),
                    'text': '- ' + el.text()
                }).appendTo("#site-navigation select");
            } else if (el.parents('.sub-menu .sub-menu').length >= 1) {
                $('<option />', {
                    'value': el.attr('href'),
                    'text': '-- ' + el.text()
                }).appendTo("#site-navigation select");
            } else {
                $('<option />', {
                    'value': el.attr('href'),
                    'text': el.text()
                }).appendTo("#site-navigation select");
            }
        });
        $("#site-navigation select").change(function() {
            window.location = $(this).find("option:selected").val();
        });

        $("#site-navigation select").uniform();
    }

    function singlePageScroll() {
        $('.menu-scroll-link a').click(function() {
            if ($('.sticky-header').width()) {
                var headerOffset = $('.sticky-header').outerHeight();
            } else {
                var headerOffset = 0;
            }
            $('body').animate({
                scrollTop: $($.attr(this, 'href')).offset().top - headerOffset
            }, 'normal');
            return false;
        });
    }

    $(document).ready(function() {
        wpexResponsiveNavTop();
        wpexMainMenuDrops();
        wpexResponsiveNav();
        wpexBackTopScroll();
        wpexCommentScroll();
        wpexPortOverlay();
        wpexLightbox();
        wpexFaqScroll();
        wpexParallaxBg();
        $(".wpex-fitvids").fitVids({
            customSelector: "iframe[src^='https://w.soundcloud.com']"
        });
        $(".fade-in-video").fadeIn();
        wpexTipsy();
        singlePageScroll();
    });
    $(window).load(function() {
        wpexFadeInSection();
        wpexFadeInElements();
        wpexTestimonialsSlider();
        wpexGalleryFormatSlider();
        if ($(window).width() > 767) {
            wpexFixedHeader();
        }
    });
});

// tipsy, facebook style tooltips for jquery
// version 1.0.0a
// (c) 2008-2010 jason frame [jason@onehackoranother.com]
// released under the MIT license
(function(e) {
    function t(e, t) {
        return typeof e == "function" ? e.call(t) : e
    }

    function n(e) {
        while (e = e.parentNode) {
            if (e == document) 
                return true
        }
        return false
    }

    function r(t, n) {
        this.$element = e(t);
        this.options = n;
        this.enabled = true;
        this.fixTitle()
    }
    r.prototype = {
        show: function() {
            var n = this.getTitle();
            if (n && this.enabled) {
                var r = this.tip();
                r.find(".tipsy-inner")[this.options.html ? "html" : "text"](n);
                r[0].className = "tipsy";
                r.remove().css({
                    top: 0,
                    left: 0,
                    visibility: "hidden",
                    display: "block"
                }).prependTo(document.body);
                var i = e.extend({}, this.$element.offset(), {
                    width: this.$element[0].offsetWidth,
                    height: this.$element[0].offsetHeight
                });
                var s = r[0].offsetWidth,
                o = r[0].offsetHeight,
                u = t(this.options.gravity, this.$element[0]);
                var a;
                switch (u.charAt(0)) {
                case "n":
                    a = {
                        top: i.top + i.height + this.options.offset,
                        left: i.left + i.width / 2 - s / 2
                    };
                    break;
                case "s":
                    a = {
                        top: i.top - o - this.options.offset,
                        left: i.left + i.width / 2 - s / 2
                    };
                    break;
                case "e":
                    a = {
                        top: i.top + i.height / 2 - o / 2,
                        left: i.left - s - this.options.offset
                    };
                    break;
                case "w":
                    a = {
                        top: i.top + i.height / 2 - o / 2,
                        left: i.left + i.width + this.options.offset
                    };
                    break
                }
                if (u.length == 2) {
                    if (u.charAt(1) == "w") {
                        a.left = i.left + i.width / 2 - 15
                    } else {
                        a.left = i.left + i.width / 2 - s + 15
                    }
                }
                r.css(a).addClass("tipsy-" + u);
                r.find(".tipsy-arrow")[0].className = "tipsy-arrow tipsy-arrow-" + u.charAt(0);
                if (this.options.className) {
                    r.addClass(t(this.options.className, this.$element[0]))
                }
                if (this.options.fade) {
                    r.stop().css({
                        opacity: 0,
                        display: "block",
                        visibility: "visible"
                    }).animate({
                        opacity: this.options.opacity
                    })
                } else {
                    r.css({
                        visibility: "visible",
                        opacity: this.options.opacity
                    })
                }
            }
        },
        hide: function() {
            if (this.options.fade) {
                this.tip().stop().fadeOut(function() {
                    e(this).remove()
                })
            } else {
                this.tip().remove()
            }
        },
        fixTitle: function() {
            var e = this.$element;
            if (e.attr("title") || typeof e.attr("original-title") != "string") {
                e.attr("original-title", e.attr("title") || "").removeAttr("title")
            }
        },
        getTitle: function() {
            var e, t = this.$element,
            n = this.options;
            this.fixTitle();
            var e, n = this.options;
            if (typeof n.title == "string") {
                e = t.attr(n.title == "title" ? "original-title" : n.title)
            } else if (typeof n.title == "function") {
                e = n.title.call(t[0])
            }
            e = ("" + e).replace(/(^\s*|\s*$)/, "");
            return e || n.fallback
        },
        tip: function() {
            if (!this.$tip) {
                this.$tip = e('<div class="tipsy"></div>').html('<div class="tipsy-arrow"></div><div class="tipsy-inner"></div>');
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
            this.enabled = !this.enabled
        }
    };
    e.fn.tipsy = function(t) {
        function i(n) {
            var i = e.data(n, "tipsy");
            if (!i) {
                i = new r(n, e.fn.tipsy.elementOptions(n, t));
                e.data(n, "tipsy", i)
            }
            return i
        }

        function s() {
            var e = i(this);
            e.hoverState = "in";
            if (t.delayIn == 0) {
                e.show()
            } else {
                e.fixTitle();
                setTimeout(function() {
                    if (e.hoverState == "in") 
                        e.show()
                }, t.delayIn)
            }
        }

        function o() {
            var e = i(this);
            e.hoverState = "out";
            if (t.delayOut == 0) {
                e.hide()
            } else {
                setTimeout(function() {
                    if (e.hoverState == "out") 
                        e.hide()
                }, t.delayOut)
            }
        }
        if (t === true) {
            return this.data("tipsy")
        } else if (typeof t == "string") {
            var n = this.data("tipsy");
            if (n) 
                n[t]();
            return this
        }
        t = e.extend({}, e.fn.tipsy.defaults, t);
        if (!t.live) 
            this.each(function() {
                i(this)
            });
        if (t.trigger != "manual") {
            var u = t.live ? "live" : "bind",
            a = t.trigger == "hover" ? "mouseenter" : "focus",
            f = t.trigger == "hover" ? "mouseleave" : "blur";
            this[u](a, s)[u](f, o)
        }
        return this
    };
    e.fn.tipsy.defaults = {
        className: null,
        delayIn: 0,
        delayOut: 0,
        fade: false,
        fallback: "",
        gravity: "n",
        html: false,
        live: false,
        offset: 0,
        opacity: .8,
        title: "title",
        trigger: "hover"
    };
    e.fn.tipsy.revalidate = function() {
        e(".tipsy").each(function() {
            var t = e.data(this, "tipsy-pointee");
            if (!t || !n(t)) {
                e(this).remove()
            }
        })
    };
    e.fn.tipsy.elementOptions = function(t, n) {
        return e.metadata ? e.extend({}, n, e(t).metadata()) : n
    };
    e.fn.tipsy.autoNS = function() {
        return e(this).offset().top > e(document).scrollTop() + e(window).height() / 2 ? "s" : "n"
    };
    e.fn.tipsy.autoWE = function() {
        return e(this).offset().left > e(document).scrollLeft() + e(window).width() / 2 ? "e" : "w"
    };
    e.fn.tipsy.autoBounds = function(t, n) {
        return function() {
            var r = {
                ns: n[0],
                ew: n.length > 1 ? n[1] : false
            }, i = e(document).scrollTop() + t,
            s = e(document).scrollLeft() + t,
            o = e(this);
            if (o.offset().top < i) 
                r.ns = "n";
            if (o.offset().left < s) 
                r.ew = "w";
            if (e(window).width() + e(document).scrollLeft() - o.offset().left < t) 
                r.ew = "e";
            if (e(window).height() + e(document).scrollTop() - o.offset().top < t) 
                r.ns = "s";
            return r.ns + (r.ew ? r.ew : "")
        }
    }
})(jQuery);



// Sticky Plugin v1.0.0 for jQuery
// =============
// Author: Anthony Garand
// Improvements by German M. Bravo (Kronuz) and Ruud Kamphuis (ruudk)
// Improvements by Leonardo C. Daronco (daronco)
// Created: 2/14/2011
// Date: 2/12/2012
// Website: http://labs.anthonygarand.com/sticky
// Description: Makes an element on the page stick on the screen as you scroll
//       It will only set the 'top' and 'position' of your element, you
//       might need to adjust the width in some cases.
(function(e) {
    var t = {
        topSpacing: 0,
        bottomSpacing: 0,
        className: "is-sticky",
        wrapperClassName: "sticky-wrapper",
        center: false,
        getWidthFrom: ""
    }, n = e(window),
    r = e(document),
    i = [],
    s = n.height(),
    o = function() {
        var t = n.scrollTop(),
        o = r.height(),
        u = o - s,
        a = t > u ? u - t : 0;
        for (var f = 0; f < i.length; f++) {
            var l = i[f],
            c = l.stickyWrapper.offset().top,
            h = c - l.topSpacing - a;
            if (t <= h) {
                if (l.currentTop !== null) {
                    l.stickyElement.css("position", "").css("top", "");
                    l.stickyElement.parent().removeClass(l.className);
                    l.currentTop = null
                }
            } else {
                var p = o - l.stickyElement.outerHeight() - l.topSpacing - l.bottomSpacing - t - a;
                if (p < 0) {
                    p = p + l.topSpacing
                } else {
                    p = l.topSpacing
                }
                if (l.currentTop != p) {
                    l.stickyElement.css("position", "fixed").css("top", p);
                    if (typeof l.getWidthFrom !== "undefined") {
                        l.stickyElement.css("width", e(l.getWidthFrom).width())
                    }
                    l.stickyElement.parent().addClass(l.className);
                    l.currentTop = p
                }
            }
        }
    }, u = function() {
        s = n.height()
    }, a = {
        init: function(n) {
            var r = e.extend(t, n);
            return this.each(function() {
                var t = e(this);
                var n = t.attr("id");
                var s = e("<div></div>").attr("id", n + "-sticky-wrapper").addClass(r.wrapperClassName);
                t.wrapAll(s);
                if (r.center) {
                    t.parent().css({
                        width: t.outerWidth(),
                        marginLeft: "auto",
                        marginRight: "auto"
                    })
                }
                if (t.css("float") == "right") {
                    t.css({
                        "float": "none"
                    }).parent().css({
                        "float": "right"
                    })
                }
                var o = t.parent();
                o.css("height", t.outerHeight());
                i.push({
                    topSpacing: r.topSpacing,
                    bottomSpacing: r.bottomSpacing,
                    stickyElement: t,
                    currentTop: null,
                    stickyWrapper: o,
                    className: r.className,
                    getWidthFrom: r.getWidthFrom
                })
            })
        },
        update: o
    };
    if (window.addEventListener) {
        window.addEventListener("scroll", o, false);
        window.addEventListener("resize", u, false)
    } else if (window.attachEvent) {
        window.attachEvent("onscroll", o);
        window.attachEvent("onresize", u)
    }
    e.fn.sticky = function(t) {
        if (a[t]) {
            return a[t].apply(this, Array.prototype.slice.call(arguments, 1))
        } else if (typeof t === "object" || !t) {
            return a.init.apply(this, arguments)
        } else {
            e.error("Method " + t + " does not exist on jQuery.sticky")
        }
    };
    e(function() {
        setTimeout(o, 0)
    })
})(jQuery);


/*
jQuery Waypoints - v2.0.3
Copyright (c) 2011-2013 Caleb Troughton
Dual licensed under the MIT license and GPL license.
https://github.com/imakewebthings/jquery-waypoints/blob/master/licenses.txt
*/
(function() {
    var t = [].indexOf || function(t) {
        for (var e = 0, n = this.length; e < n; e++) {
            if (e in this && this[e] === t) 
                return e
        }
        return - 1
    }, e = [].slice;
    (function(t, e) {
        if (typeof define === "function" && define.amd) {
            return define("waypoints", ["jquery"], function(n) {
                return e(n, t)
            })
        } else {
            return e(t.jQuery, t)
        }
    })(this, function(n, r) {
        var i, o, l, s, f, u, a, c, h, d, p, y, v, w, g, m;
        i = n(r);
        c = t.call(r, "ontouchstart") >= 0;
        s = {
            horizontal: {},
            vertical: {}
        };
        f = 1;
        a = {};
        u = "waypoints-context-id";
        p = "resize.waypoints";
        y = "scroll.waypoints";
        v = 1;
        w = "waypoints-waypoint-ids";
        g = "waypoint";
        m = "waypoints";
        o = function() {
            function t(t) {
                var e = this;
                this.$element = t;
                this.element = t[0];
                this.didResize = false;
                this.didScroll = false;
                this.id = "context" + f++;
                this.oldScroll = {
                    x: t.scrollLeft(),
                    y: t.scrollTop()
                };
                this.waypoints = {
                    horizontal: {},
                    vertical: {}
                };
                t.data(u, this.id);
                a[this.id] = this;
                t.bind(y, function() {
                    var t;
                    if (!(e.didScroll || c)) {
                        e.didScroll = true;
                        t = function() {
                            e.doScroll();
                            return e.didScroll = false
                        };
                        return r.setTimeout(t, n[m].settings.scrollThrottle)
                    }
                });
                t.bind(p, function() {
                    var t;
                    if (!e.didResize) {
                        e.didResize = true;
                        t = function() {
                            n[m]("refresh");
                            return e.didResize = false
                        };
                        return r.setTimeout(t, n[m].settings.resizeThrottle)
                    }
                })
            }
            t.prototype.doScroll = function() {
                var t, e = this;
                t = {
                    horizontal: {
                        newScroll: this.$element.scrollLeft(),
                        oldScroll: this.oldScroll.x,
                        forward: "right",
                        backward: "left"
                    },
                    vertical: {
                        newScroll: this.$element.scrollTop(),
                        oldScroll: this.oldScroll.y,
                        forward: "down",
                        backward: "up"
                    }
                };
                if (c && (!t.vertical.oldScroll || !t.vertical.newScroll)) {
                    n[m]("refresh")
                }
                n.each(t, function(t, r) {
                    var i, o, l;
                    l = [];
                    o = r.newScroll > r.oldScroll;
                    i = o ? r.forward : r.backward;
                    n.each(e.waypoints[t], function(t, e) {
                        var n, i;
                        if (r.oldScroll < (n = e.offset) && n <= r.newScroll) {
                            return l.push(e)
                        } else if (r.newScroll < (i = e.offset) && i <= r.oldScroll) {
                            return l.push(e)
                        }
                    });
                    l.sort(function(t, e) {
                        return t.offset - e.offset
                    });
                    if (!o) {
                        l.reverse()
                    }
                    return n.each(l, function(t, e) {
                        if (e.options.continuous || t === l.length - 1) {
                            return e.trigger([i])
                        }
                    })
                });
                return this.oldScroll = {
                    x: t.horizontal.newScroll,
                    y: t.vertical.newScroll
                }
            };
            t.prototype.refresh = function() {
                var t, e, r, i = this;
                r = n.isWindow(this.element);
                e = this.$element.offset();
                this.doScroll();
                t = {
                    horizontal: {
                        contextOffset: r ? 0 : e.left,
                        contextScroll: r ? 0 : this.oldScroll.x,
                        contextDimension: this.$element.width(),
                        oldScroll: this.oldScroll.x,
                        forward: "right",
                        backward: "left",
                        offsetProp: "left"
                    },
                    vertical: {
                        contextOffset: r ? 0 : e.top,
                        contextScroll: r ? 0 : this.oldScroll.y,
                        contextDimension: r ? n[m]("viewportHeight") : this.$element.height(),
                        oldScroll: this.oldScroll.y,
                        forward: "down",
                        backward: "up",
                        offsetProp: "top"
                    }
                };
                return n.each(t, function(t, e) {
                    return n.each(i.waypoints[t], function(t, r) {
                        var i, o, l, s, f;
                        i = r.options.offset;
                        l = r.offset;
                        o = n.isWindow(r.element) ? 0 : r.$element.offset()[e.offsetProp];
                        if (n.isFunction(i)) {
                            i = i.apply(r.element)
                        } else if (typeof i === "string") {
                            i = parseFloat(i);
                            if (r.options.offset.indexOf("%") > - 1) {
                                i = Math.ceil(e.contextDimension * i / 100)
                            }
                        }
                        r.offset = o - e.contextOffset + e.contextScroll - i;
                        if (r.options.onlyOnScroll && l != null || !r.enabled) {
                            return
                        }
                        if (l !== null && l < (s = e.oldScroll) && s <= r.offset) {
                            return r.trigger([e.backward])
                        } else if (l !== null && l > (f = e.oldScroll) && f >= r.offset) {
                            return r.trigger([e.forward])
                        } else if (l === null && e.oldScroll >= r.offset) {
                            return r.trigger([e.forward])
                        }
                    })
                })
            };
            t.prototype.checkEmpty = function() {
                if (n.isEmptyObject(this.waypoints.horizontal) && n.isEmptyObject(this.waypoints.vertical)) {
                    this.$element.unbind([p, y].join(" "));
                    return delete a[this.id]
                }
            };
            return t
        }();
        l = function() {
            function t(t, e, r) {
                var i, o;
                r = n.extend({}, n.fn[g].defaults, r);
                if (r.offset === "bottom-in-view") {
                    r.offset = function() {
                        var t;
                        t = n[m]("viewportHeight");
                        if (!n.isWindow(e.element)) {
                            t = e.$element.height()
                        }
                        return t - n(this).outerHeight()
                    }
                }
                this.$element = t;
                this.element = t[0];
                this.axis = r.horizontal ? "horizontal" : "vertical";
                this.callback = r.handler;
                this.context = e;
                this.enabled = r.enabled;
                this.id = "waypoints" + v++;
                this.offset = null;
                this.options = r;
                e.waypoints[this.axis][this.id] = this;
                s[this.axis][this.id] = this;
                i = (o = t.data(w)) != null ? o : [];
                i.push(this.id);
                t.data(w, i)
            }
            t.prototype.trigger = function(t) {
                if (!this.enabled) {
                    return
                }
                if (this.callback != null) {
                    this.callback.apply(this.element, t)
                }
                if (this.options.triggerOnce) {
                    return this.destroy()
                }
            };
            t.prototype.disable = function() {
                return this.enabled = false
            };
            t.prototype.enable = function() {
                this.context.refresh();
                return this.enabled = true
            };
            t.prototype.destroy = function() {
                delete s[this.axis][this.id];
                delete this.context.waypoints[this.axis][this.id];
                return this.context.checkEmpty()
            };
            t.getWaypointsByElement = function(t) {
                var e, r;
                r = n(t).data(w);
                if (!r) {
                    return []
                }
                e = n.extend({}, s.horizontal, s.vertical);
                return n.map(r, function(t) {
                    return e[t]
                })
            };
            return t
        }();
        d = {
            init: function(t, e) {
                var r;
                if (e == null) {
                    e = {}
                }
                if ((r = e.handler) == null) {
                    e.handler = t
                }
                this.each(function() {
                    var t, r, i, s;
                    t = n(this);
                    i = (s = e.context) != null ? s : n.fn[g].defaults.context;
                    if (!n.isWindow(i)) {
                        i = t.closest(i)
                    }
                    i = n(i);
                    r = a[i.data(u)];
                    if (!r) {
                        r = new o(i)
                    }
                    return new l(t, r, e)
                });
                n[m]("refresh");
                return this
            },
            disable: function() {
                return d._invoke(this, "disable")
            },
            enable: function() {
                return d._invoke(this, "enable")
            },
            destroy: function() {
                return d._invoke(this, "destroy")
            },
            prev: function(t, e) {
                return d._traverse.call(this, t, e, function(t, e, n) {
                    if (e > 0) {
                        return t.push(n[e - 1])
                    }
                })
            },
            next: function(t, e) {
                return d._traverse.call(this, t, e, function(t, e, n) {
                    if (e < n.length - 1) {
                        return t.push(n[e + 1])
                    }
                })
            },
            _traverse: function(t, e, i) {
                var o, l;
                if (t == null) {
                    t = "vertical"
                }
                if (e == null) {
                    e = r
                }
                l = h.aggregate(e);
                o = [];
                this.each(function() {
                    var e;
                    e = n.inArray(this, l[t]);
                    return i(o, e, l[t])
                });
                return this.pushStack(o)
            },
            _invoke: function(t, e) {
                t.each(function() {
                    var t;
                    t = l.getWaypointsByElement(this);
                    return n.each(t, function(t, n) {
                        n[e]();
                        return true
                    })
                });
                return this
            }
        };
        n.fn[g] = function() {
            var t, r;
            r = arguments[0], t = 2 <= arguments.length ? e.call(arguments, 1) : [];
            if (d[r]) {
                return d[r].apply(this, t)
            } else if (n.isFunction(r)) {
                return d.init.apply(this, arguments)
            } else if (n.isPlainObject(r)) {
                return d.init.apply(this, [null, r])
            } else if (!r) {
                return n.error("jQuery Waypoints needs a callback function or handler option.")
            } else {
                return n.error("The " + r + " method does not exist in jQuery Waypoints.")
            }
        };
        n.fn[g].defaults = {
            context: r,
            continuous: true,
            enabled: true,
            horizontal: false,
            offset: 0,
            triggerOnce: false
        };
        h = {
            refresh: function() {
                return n.each(a, function(t, e) {
                    return e.refresh()
                })
            },
            viewportHeight: function() {
                var t;
                return (t = r.innerHeight) != null ? t : i.height()
            },
            aggregate: function(t) {
                var e, r, i;
                e = s;
                if (t) {
                    e = (i = a[n(t).data(u)]) != null ? i.waypoints : void 0
                }
                if (!e) {
                    return []
                }
                r = {
                    horizontal: [],
                    vertical: []
                };
                n.each(r, function(t, i) {
                    n.each(e[t], function(t, e) {
                        return i.push(e)
                    });
                    i.sort(function(t, e) {
                        return t.offset - e.offset
                    });
                    r[t] = n.map(i, function(t) {
                        return t.element
                    });
                    return r[t] = n.unique(r[t])
                });
                return r
            },
            above: function(t) {
                if (t == null) {
                    t = r
                }
                return h._filter(t, "vertical", function(t, e) {
                    return e.offset <= t.oldScroll.y
                })
            },
            below: function(t) {
                if (t == null) {
                    t = r
                }
                return h._filter(t, "vertical", function(t, e) {
                    return e.offset > t.oldScroll.y
                })
            },
            left: function(t) {
                if (t == null) {
                    t = r
                }
                return h._filter(t, "horizontal", function(t, e) {
                    return e.offset <= t.oldScroll.x
                })
            },
            right: function(t) {
                if (t == null) {
                    t = r
                }
                return h._filter(t, "horizontal", function(t, e) {
                    return e.offset > t.oldScroll.x
                })
            },
            enable: function() {
                return h._invoke("enable")
            },
            disable: function() {
                return h._invoke("disable")
            },
            destroy: function() {
                return h._invoke("destroy")
            },
            extendFn: function(t, e) {
                return d[t] = e
            },
            _invoke: function(t) {
                var e;
                e = n.extend({}, s.vertical, s.horizontal);
                return n.each(e, function(e, n) {
                    n[t]();
                    return true
                })
            },
            _filter: function(t, e, r) {
                var i, o;
                i = a[n(t).data(u)];
                if (!i) {
                    return []
                }
                o = [];
                n.each(i.waypoints[e], function(t, e) {
                    if (r(i, e)) {
                        return o.push(e)
                    }
                });
                o.sort(function(t, e) {
                    return t.offset - e.offset
                });
                return n.map(o, function(t) {
                    return t.element
                })
            }
        };
        n[m] = function() {
            var t, n;
            n = arguments[0], t = 2 <= arguments.length ? e.call(arguments, 1) : [];
            if (h[n]) {
                return h[n].apply(null, t)
            } else {
                return h.aggregate.call(null, n)
            }
        };
        n[m].settings = {
            resizeThrottle: 100,
            scrollThrottle: 30
        };
        return i.load(function() {
            return n[m]("refresh")
        })
    })
}).call(this);


// Generated by CoffeeScript 1.4.0
/*
Sticky Elements Shortcut for jQuery Waypoints - v2.0.2
Copyright (c) 2011-2013 Caleb Troughton
Dual licensed under the MIT license and GPL license.
https://github.com/imakewebthings/jquery-waypoints/blob/master/licenses.txt
*/
(function() {
    (function(t, n) {
        if (typeof define === "function" && define.amd) {
            return define(["jquery", "waypoints"], n)
        } else {
            return n(t.jQuery)
        }
    })(this, function(t) {
        var n, e;
        n = {
            wrapper: '<div class="sticky-wrapper" />',
            stuckClass: "stuck"
        };
        e = function(n, e) {
            n.wrap(e.wrapper);
            n.each(function() {
                var n;
                n = t(this);
                n.parent().height(n.outerHeight());
                return true
            });
            return n.parent()
        };
        return t.waypoints("extendFn", "sticky", function(r) {
            var i, a;
            r = t.extend({}, t.fn.waypoint.defaults, n, r);
            i = e(this, r);
            a = r.handler;
            r.handler = function(n) {
                var e, i;
                e = t(this).children(":first");
                i = n === "down" || n === "right";
                e.toggleClass(r.stuckClass, i);
                if (a != null) {
                    return a.call(this, n)
                }
            };
            i.waypoint(r);
            return this
        })
    })
}).call(this);

/*
Uniform v2.1.0
Copyright © 2009 Josh Pyles / Pixelmatrix Design LLC
http://pixelmatrixdesign.com
Requires jQuery 1.3 or newer
Much thanks to Thomas Reynolds and Buck Wilson for their help and advice on
this.
Disabling text selection is made possible by Mathias Bynens
<http://mathiasbynens.be/> and his noSelect plugin.
<https://github.com/mathiasbynens/jquery-noselect>, which is embedded.
Also, thanks to David Kaneda and Eugene Bond for their contributions to the
plugin.
Tyler Akins has also rewritten chunks of the plugin, helped close many issues,
and ensured version 2 got out the door.
License:
MIT License - http://www.opensource.org/licenses/mit-license.php
Enjoy!
*/
/*global jQuery, window, document, navigator*/
(function(a) {
    a.uniform = {
        options: {
            selectClass: "selector",
            radioClass: "radio",
            checkboxClass: "checker",
            fileClass: "uploader",
            filenameClass: "filename",
            fileBtnClass: "action",
            fileDefaultText: "No file selected",
            fileBtnText: "Choose File",
            checkedClass: "checked",
            focusClass: "focus",
            disabledClass: "disabled",
            buttonClass: "button",
            activeClass: "active",
            hoverClass: "hover",
            useID: true,
            idPrefix: "uniform",
            resetSelector: false,
            autoHide: true
        },
        elements: []
    };
    if (a.browser.msie && a.browser.version < 7) {
        a.support.selectOpacity = false
    } else {
        a.support.selectOpacity = true
    }
    a.fn.uniform = function(k) {
        k = a.extend(a.uniform.options, k);
        var d = this;
        if (k.resetSelector != false) {
            a(k.resetSelector).mouseup(function() {
                function l() {
                    a.uniform.update(d)
                }
                setTimeout(l, 10)
            })
        }

        function j(l) {
            $el = a(l);
            $el.addClass($el.attr("type"));
            b(l)
        }

        function g(l) {
            a(l).addClass("uniform");
            b(l)
        }

        function i(o) {
            var m = a(o);
            var p = a("<div>"),
            l = a("<span>");
            p.addClass(k.buttonClass);
            if (k.useID && m.attr("id") != "") {
                p.attr("id", k.idPrefix + "-" + m.attr("id"))
            }
            var n;
            if (m.is("a") || m.is("button")) {
                n = m.text()
            } else {
                if (m.is(":submit") || m.is(":reset") || m.is("input[type=button]")) {
                    n = m.attr("value")
                }
            }
            n = n == "" ? m.is(":reset") ? "Reset" : "Submit" : n;
            l.html(n);
            m.css("opacity", 0);
            m.wrap(p);
            m.wrap(l);
            p = m.closest("div");
            l = m.closest("span");
            if (m.is(":disabled")) {
                p.addClass(k.disabledClass)
            }
            p.bind({
                "mouseenter.uniform": function() {
                    p.addClass(k.hoverClass)
                },
                "mouseleave.uniform": function() {
                    p.removeClass(k.hoverClass);
                    p.removeClass(k.activeClass)
                },
                "mousedown.uniform touchbegin.uniform": function() {
                    p.addClass(k.activeClass)
                },
                "mouseup.uniform touchend.uniform": function() {
                    p.removeClass(k.activeClass)
                },
                "click.uniform touchend.uniform": function(r) {
                    if (a(r.target).is("span") || a(r.target).is("div")) {
                        if (o[0].dispatchEvent) {
                            var q = document.createEvent("MouseEvents");
                            q.initEvent("click", true, true);
                            o[0].dispatchEvent(q)
                        } else {
                            o[0].click()
                        }
                    }
                }
            });
            o.bind({
                "focus.uniform": function() {
                    p.addClass(k.focusClass)
                },
                "blur.uniform": function() {
                    p.removeClass(k.focusClass)
                }
            });
            a.uniform.noSelect(p);
            b(o)
        }

        function e(o) {
            var m = a(o);
            var p = a("<div />"),
            l = a("<span />");
            if (!m.css("display") == "none" && k.autoHide) {
                p.hide()
            }
            p.addClass(k.selectClass);
            if (k.useID && o.attr("id") != "") {
                p.attr("id", k.idPrefix + "-" + o.attr("id"))
            }
            var n = o.find(":selected:first");
            if (n.length == 0) {
                n = o.find("option:first")
            }
            l.html(n.html());
            o.css("opacity", 0);
            o.wrap(p);
            o.before(l);
            p = o.parent("div");
            l = o.siblings("span");
            o.bind({
                "change.uniform": function() {
                    l.text(o.find(":selected").html());
                    p.removeClass(k.activeClass)
                },
                "focus.uniform": function() {
                    p.addClass(k.focusClass)
                },
                "blur.uniform": function() {
                    p.removeClass(k.focusClass);
                    p.removeClass(k.activeClass)
                },
                "mousedown.uniform touchbegin.uniform": function() {
                    p.addClass(k.activeClass)
                },
                "mouseup.uniform touchend.uniform": function() {
                    p.removeClass(k.activeClass)
                },
                "click.uniform touchend.uniform": function() {
                    p.removeClass(k.activeClass)
                },
                "mouseenter.uniform": function() {
                    p.addClass(k.hoverClass)
                },
                "mouseleave.uniform": function() {
                    p.removeClass(k.hoverClass);
                    p.removeClass(k.activeClass)
                },
                "keyup.uniform": function() {
                    l.text(o.find(":selected").html())
                }
            });
            if (a(o).attr("disabled")) {
                p.addClass(k.disabledClass)
            }
            a.uniform.noSelect(l);
            b(o)
        }

        function f(n) {
            var m = a(n);
            var o = a("<div />"),
            l = a("<span />");
            if (!m.css("display") == "none" && k.autoHide) {
                o.hide()
            }
            o.addClass(k.checkboxClass);
            if (k.useID && n.attr("id") != "") {
                o.attr("id", k.idPrefix + "-" + n.attr("id"))
            }
            a(n).wrap(o);
            a(n).wrap(l);
            l = n.parent();
            o = l.parent();
            a(n).css("opacity", 0).bind({
                "focus.uniform": function() {
                    o.addClass(k.focusClass)
                },
                "blur.uniform": function() {
                    o.removeClass(k.focusClass)
                },
                "click.uniform touchend.uniform": function() {
                    if (!a(n).attr("checked")) {
                        l.removeClass(k.checkedClass)
                    } else {
                        l.addClass(k.checkedClass)
                    }
                },
                "mousedown.uniform touchbegin.uniform": function() {
                    o.addClass(k.activeClass)
                },
                "mouseup.uniform touchend.uniform": function() {
                    o.removeClass(k.activeClass)
                },
                "mouseenter.uniform": function() {
                    o.addClass(k.hoverClass)
                },
                "mouseleave.uniform": function() {
                    o.removeClass(k.hoverClass);
                    o.removeClass(k.activeClass)
                }
            });
            if (a(n).attr("checked")) {
                l.addClass(k.checkedClass)
            }
            if (a(n).attr("disabled")) {
                o.addClass(k.disabledClass)
            }
            b(n)
        }

        function c(n) {
            var m = a(n);
            var o = a("<div />"),
            l = a("<span />");
            if (!m.css("display") == "none" && k.autoHide) {
                o.hide()
            }
            o.addClass(k.radioClass);
            if (k.useID && n.attr("id") != "") {
                o.attr("id", k.idPrefix + "-" + n.attr("id"))
            }
            a(n).wrap(o);
            a(n).wrap(l);
            l = n.parent();
            o = l.parent();
            a(n).css("opacity", 0).bind({
                "focus.uniform": function() {
                    o.addClass(k.focusClass)
                },
                "blur.uniform": function() {
                    o.removeClass(k.focusClass)
                },
                "click.uniform touchend.uniform": function() {
                    if (!a(n).attr("checked")) {
                        l.removeClass(k.checkedClass)
                    } else {
                        var p = k.radioClass.split(" ")[0];
                        a("." + p + " span." + k.checkedClass + ":has([name='" + a(n).attr("name") + "'])").removeClass(k.checkedClass);
                        l.addClass(k.checkedClass)
                    }
                },
                "mousedown.uniform touchend.uniform": function() {
                    if (!a(n).is(":disabled")) {
                        o.addClass(k.activeClass)
                    }
                },
                "mouseup.uniform touchbegin.uniform": function() {
                    o.removeClass(k.activeClass)
                },
                "mouseenter.uniform touchend.uniform": function() {
                    o.addClass(k.hoverClass)
                },
                "mouseleave.uniform": function() {
                    o.removeClass(k.hoverClass);
                    o.removeClass(k.activeClass)
                }
            });
            if (a(n).attr("checked")) {
                l.addClass(k.checkedClass)
            }
            if (a(n).attr("disabled")) {
                o.addClass(k.disabledClass)
            }
            b(n)
        }

        function h(q) {
            var o = a(q);
            var r = a("<div />"),
            p = a("<span>" + k.fileDefaultText + "</span>"),
            m = a("<span>" + k.fileBtnText + "</span>");
            if (!o.css("display") == "none" && k.autoHide) {
                r.hide()
            }
            r.addClass(k.fileClass);
            p.addClass(k.filenameClass);
            m.addClass(k.fileBtnClass);
            if (k.useID && o.attr("id") != "") {
                r.attr("id", k.idPrefix + "-" + o.attr("id"))
            }
            o.wrap(r);
            o.after(m);
            o.after(p);
            r = o.closest("div");
            p = o.siblings("." + k.filenameClass);
            m = o.siblings("." + k.fileBtnClass);
            if (!o.attr("size")) {
                var l = r.width();
                o.attr("size", l / 10)
            }
            var n = function() {
                var s = o.val();
                if (s === "") {
                    s = k.fileDefaultText
                } else {
                    s = s.split(/[\/\\]+/);
                    s = s[(s.length - 1)]
                }
                p.text(s)
            };
            n();
            o.css("opacity", 0).bind({
                "focus.uniform": function() {
                    r.addClass(k.focusClass)
                },
                "blur.uniform": function() {
                    r.removeClass(k.focusClass)
                },
                "mousedown.uniform": function() {
                    if (!a(q).is(":disabled")) {
                        r.addClass(k.activeClass)
                    }
                },
                "mouseup.uniform": function() {
                    r.removeClass(k.activeClass)
                },
                "mouseenter.uniform": function() {
                    r.addClass(k.hoverClass)
                },
                "mouseleave.uniform": function() {
                    r.removeClass(k.hoverClass);
                    r.removeClass(k.activeClass)
                }
            });
            if (a.browser.msie) {
                o.bind("click.uniform.ie7", function() {
                    setTimeout(n, 0)
                })
            } else {
                o.bind("change.uniform", n)
            }
            if (o.attr("disabled")) {
                r.addClass(k.disabledClass)
            }
            a.uniform.noSelect(p);
            a.uniform.noSelect(m);
            b(q)
        }
        a.uniform.restore = function(l) {
            if (l == undefined) {
                l = a(a.uniform.elements)
            }
            a(l).each(function() {
                if (a(this).is(":checkbox")) {
                    a(this).unwrap().unwrap()
                } else {
                    if (a(this).is("select")) {
                        a(this).siblings("span").remove();
                        a(this).unwrap()
                    } else {
                        if (a(this).is(":radio")) {
                            a(this).unwrap().unwrap()
                        } else {
                            if (a(this).is(":file")) {
                                a(this).siblings("span").remove();
                                a(this).unwrap()
                            } else {
                                if (a(this).is("button, :submit, :reset, a, input[type='button']")) {
                                    a(this).unwrap().unwrap()
                                }
                            }
                        }
                    }
                }
                a(this).unbind(".uniform");
                a(this).css("opacity", "1");
                var m = a.inArray(a(l), a.uniform.elements);
                a.uniform.elements.splice(m, 1)
            })
        };

        function b(l) {
            l = a(l).get();
            if (l.length > 1) {
                a.each(l, function(m, n) {
                    a.uniform.elements.push(n)
                })
            } else {
                a.uniform.elements.push(l)
            }
        }
        a.uniform.noSelect = function(l) {
            function m() {
                return false
            }
            a(l).each(function() {
                this.onselectstart = this.ondragstart = m;
                a(this).mousedown(m).css({
                    MozUserSelect: "none"
                })
            })
        };
        a.uniform.update = function(l) {
            if (l == undefined) {
                l = a(a.uniform.elements)
            }
            l = a(l);
            l.each(function() {
                var n = a(this);
                if (n.is("select")) {
                    var m = n.siblings("span");
                    var p = n.parent("div");
                    p.removeClass(k.hoverClass + " " + k.focusClass + " " + k.activeClass);
                    m.html(n.find(":selected").html());
                    if (n.is(":disabled")) {
                        p.addClass(k.disabledClass)
                    } else {
                        p.removeClass(k.disabledClass)
                    }
                } else {
                    if (n.is(":checkbox")) {
                        var m = n.closest("span");
                        var p = n.closest("div");
                        p.removeClass(k.hoverClass + " " + k.focusClass + " " + k.activeClass);
                        m.removeClass(k.checkedClass);
                        if (n.is(":checked")) {
                            m.addClass(k.checkedClass)
                        }
                        if (n.is(":disabled")) {
                            p.addClass(k.disabledClass)
                        } else {
                            p.removeClass(k.disabledClass)
                        }
                    } else {
                        if (n.is(":radio")) {
                            var m = n.closest("span");
                            var p = n.closest("div");
                            p.removeClass(k.hoverClass + " " + k.focusClass + " " + k.activeClass);
                            m.removeClass(k.checkedClass);
                            if (n.is(":checked")) {
                                m.addClass(k.checkedClass)
                            }
                            if (n.is(":disabled")) {
                                p.addClass(k.disabledClass)
                            } else {
                                p.removeClass(k.disabledClass)
                            }
                        } else {
                            if (n.is(":file")) {
                                var p = n.parent("div");
                                var o = n.siblings(k.filenameClass);
                                btnTag = n.siblings(k.fileBtnClass);
                                p.removeClass(k.hoverClass + " " + k.focusClass + " " + k.activeClass);
                                o.text(n.val());
                                if (n.is(":disabled")) {
                                    p.addClass(k.disabledClass)
                                } else {
                                    p.removeClass(k.disabledClass)
                                }
                            } else {
                                if (n.is(":submit") || n.is(":reset") || n.is("button") || n.is("a") || l.is("input[type=button]")) {
                                    var p = n.closest("div");
                                    p.removeClass(k.hoverClass + " " + k.focusClass + " " + k.activeClass);
                                    if (n.is(":disabled")) {
                                        p.addClass(k.disabledClass)
                                    } else {
                                        p.removeClass(k.disabledClass)
                                    }
                                }
                            }
                        }
                    }
                }
            })
        };
        return this.each(function() {
            if (a.support.selectOpacity) {
                var l = a(this);
                if (l.is("select")) {
                    if (l.attr("multiple") != true) {
                        if (l.attr("size") == undefined || l.attr("size") <= 1) {
                            e(l)
                        }
                    }
                } else {
                    if (l.is(":checkbox")) {
                        f(l)
                    } else {
                        if (l.is(":radio")) {
                            c(l)
                        } else {
                            if (l.is(":file")) {
                                h(l)
                            } else {
                                if (l.is(":text, :password, input[type='email']")) {
                                    j(l)
                                } else {
                                    if (l.is("textarea")) {
                                        g(l)
                                    } else {
                                        if (l.is("a") || l.is(":submit") || l.is(":reset") || l.is("button") || l.is("input[type=button]")) {
                                            i(l)
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        })
    }
})(jQuery);


/*! Magnific Popup - v0.9.3 - 2013-07-16
 * http://dimsemenov.com/plugins/magnific-popup/
 * Copyright (c) 2013 Dmitry Semenov; */
(function(e) {
    var t, i, n, o, a, r, s, l = "Close",
    c = "BeforeClose",
    d = "AfterClose",
    u = "BeforeAppend",
    p = "MarkupParse",
    f = "Open",
    m = "Change",
    g = "mfp",
    v = "." + g,
    h = "mfp-ready",
    C = "mfp-removing",
    y = "mfp-prevent-close",
    w = function() {}, b = !! window.jQuery,
    I = e(window),
    x = function(e, i) {
        t.ev.on(g + e + v, i)
    }, k = function(t, i, n, o) {
        var a = document.createElement("div");
        return a.className = "mfp-" + t, n && (a.innerHTML = n), o ? i && i.appendChild(a) : (a = e(a), i && a.appendTo(i)), a
    }, T = function(i, n) {
        t.ev.triggerHandler(g + i, n), t.st.callbacks && (i = i.charAt(0).toLowerCase() + i.slice(1), t.st.callbacks[i] && t.st.callbacks[i].apply(t, e.isArray(n) ? n : [n]))
    }, E = function() {
        (t.st.focus ? t.content.find(t.st.focus).eq(0) : t.wrap).trigger("focus")
    }, S = function(i) {
        return i === s && t.currTemplate.closeBtn || (t.currTemplate.closeBtn = e(t.st.closeMarkup.replace("%title%", t.st.tClose)), s = i), t.currTemplate.closeBtn
    }, P = function() {
        e.magnificPopup.instance || (t = new w, t.init(), e.magnificPopup.instance = t)
    }, _ = function(i) {
        if (!e(i).hasClass(y)) {
            var n = t.st.closeOnContentClick,
            o = t.st.closeOnBgClick;
            if (n && o) 
                return !0;
            if (!t.content || e(i).hasClass("mfp-close") || t.preloader && i === t.preloader[0]) 
                return !0;
            if (i === t.content[0] || e.contains(t.content[0], i)) {
                if (n) 
                    return !0
            } else if (o && e.contains(document, i)) 
                return !0;
            return !1
        }
    }, O = function() {
        var e = document.createElement("p").style,
        t = ["ms", "O", "Moz", "Webkit"];
        if (void 0 !== e.transition) 
            return !0;
        for (; t.length;)
            if (t.pop() + "Transition" in e) 
                return !0;
        return !1
    };
    w.prototype = {
        constructor: w,
        init: function() {
            var i = navigator.appVersion;
            t.isIE7 = - 1 !== i.indexOf("MSIE 7."), t.isIE8 = - 1 !== i.indexOf("MSIE 8."), t.isLowIE = t.isIE7 || t.isIE8, t.isAndroid = /android/gi.test(i), t.isIOS = /iphone|ipad|ipod/gi.test(i), t.supportsTransition = O(), t.probablyMobile = t.isAndroid || t.isIOS || /(Opera Mini)|Kindle|webOS|BlackBerry|(Opera Mobi)|(Windows Phone)|IEMobile/i.test(navigator.userAgent), n = e(document.body), o = e(document), t.popupsCache = {}
        },
        open: function(i) {
            var n;
            if (i.isObj === !1) {
                t.items = i.items.toArray(), t.index = 0;
                var a, s = i.items;
                for (n = 0; s.length > n; n++)
                    if (a = s[n], a.parsed && (a = a.el[0]), a === i.el[0]) {
                        t.index = n;
                        break
                    }
            } else 
                t.items = e.isArray(i.items) ? i.items : [i.items], t.index = i.index || 0;
            if (t.isOpen) 
                return t.updateItemHTML(), void 0;
            t.types = [], r = "", t.ev = i.mainEl && i.mainEl.length ? i.mainEl.eq(0) : o, i.key ? (t.popupsCache[i.key] || (t.popupsCache[i.key] = {}), t.currTemplate = t.popupsCache[i.key]) : t.currTemplate = {}, t.st = e.extend(!0, {}, e.magnificPopup.defaults, i), t.fixedContentPos = "auto" === t.st.fixedContentPos ? !t.probablyMobile : t.st.fixedContentPos, t.st.modal && (t.st.closeOnContentClick = !1, t.st.closeOnBgClick = !1, t.st.showCloseBtn = !1, t.st.enableEscapeKey = !1), t.bgOverlay || (t.bgOverlay = k("bg").on("click" + v, function() {
                t.close()
            }), t.wrap = k("wrap").attr("tabindex", - 1).on("click" + v, function(e) {
                _(e.target) && t.close()
            }), t.container = k("container", t.wrap)), t.contentContainer = k("content"), t.st.preloader && (t.preloader = k("preloader", t.container, t.st.tLoading));
            var l = e.magnificPopup.modules;
            for (n = 0; l.length > n; n++) {
                var c = l[n];
                c = c.charAt(0).toUpperCase() + c.slice(1), t["init" + c].call(t)
            }
            T("BeforeOpen"), t.st.showCloseBtn && (t.st.closeBtnInside ? (x(p, function(e, t, i, n) {
                i.close_replaceWith = S(n.type)
            }), r += " mfp-close-btn-in") : t.wrap.append(S())), t.st.alignTop && (r += " mfp-align-top"), t.fixedContentPos ? t.wrap.css({
                overflow: t.st.overflowY,
                overflowX: "hidden",
                overflowY: t.st.overflowY
            }) : t.wrap.css({
                top: I.scrollTop(),
                position: "absolute"
            }), (t.st.fixedBgPos === !1 || "auto" === t.st.fixedBgPos && !t.fixedContentPos) && t.bgOverlay.css({
                height: o.height(),
                position: "absolute"
            }), t.st.enableEscapeKey && o.on("keyup" + v, function(e) {
                27 === e.keyCode && t.close()
            }), I.on("resize" + v, function() {
                t.updateSize()
            }), t.st.closeOnContentClick || (r += " mfp-auto-cursor"), r && t.wrap.addClass(r);
            var d = t.wH = I.height(),
            u = {};
            if (t.fixedContentPos && t._hasScrollBar(d)) {
                var m = t._getScrollbarSize();
                m && (u.paddingRight = m)
            }
            t.fixedContentPos && (t.isIE7 ? e("body, html").css("overflow", "hidden") : u.overflow = "hidden");
            var g = t.st.mainClass;
            t.isIE7 && (g += " mfp-ie7"), g && t._addClassToMFP(g), t.updateItemHTML(), T("BuildControls"), e("html").css(u), t.bgOverlay.add(t.wrap).prependTo(document.body), t._lastFocusedEl = document.activeElement, setTimeout(function() {
                t.content ? (t._addClassToMFP(h), E()) : t.bgOverlay.addClass(h), o.on("focusin" + v, function(i) {
                    return i.target === t.wrap[0] || e.contains(t.wrap[0], i.target) ? void 0 : (E(), !1)
                })
            }, 16), t.isOpen = !0, t.updateSize(d), T(f)
        },
        close: function() {
            t.isOpen && (T(c), t.isOpen = !1, t.st.removalDelay && !t.isLowIE && t.supportsTransition ? (t._addClassToMFP(C), setTimeout(function() {
                t._close()
            }, t.st.removalDelay)) : t._close())
        },
        _close: function() {
            T(l);
            var i = C + " " + h + " ";
            if (t.bgOverlay.detach(), t.wrap.detach(), t.container.empty(), t.st.mainClass && (i += t.st.mainClass + " "), t._removeClassFromMFP(i), t.fixedContentPos) {
                var n = {
                    paddingRight: ""
                };
                t.isIE7 ? e("body, html").css("overflow", "") : n.overflow = "", e("html").css(n)
            }
            o.off("keyup" + v + " focusin" + v), t.ev.off(v), t.wrap.attr("class", "mfp-wrap").removeAttr("style"), t.bgOverlay.attr("class", "mfp-bg"), t.container.attr("class", "mfp-container"), !t.st.showCloseBtn || t.st.closeBtnInside && t.currTemplate[t.currItem.type] !== !0 || t.currTemplate.closeBtn && t.currTemplate.closeBtn.detach(), t._lastFocusedEl && e(t._lastFocusedEl).trigger("focus"), t.currItem = null, t.content = null, t.currTemplate = null, t.prevHeight = 0, T(d)
        },
        updateSize: function(e) {
            if (t.isIOS) {
                var i = document.documentElement.clientWidth / window.innerWidth,
                n = window.innerHeight * i;
                t.wrap.css("height", n), t.wH = n
            } else 
                t.wH = e || I.height();
            t.fixedContentPos || t.wrap.css("height", t.wH), T("Resize")
        },
        updateItemHTML: function() {
            var i = t.items[t.index];
            t.contentContainer.detach(), t.content && t.content.detach(), i.parsed || (i = t.parseEl(t.index));
            var n = i.type;
            if (T("BeforeChange", [t.currItem ? t.currItem.type : "", n]), t.currItem = i, !t.currTemplate[n]) {
                var o = t.st[n] ? t.st[n].markup : !1;
                T("FirstMarkupParse", o), t.currTemplate[n] = o ? e(o) : !0
            }
            a && a !== i.type && t.container.removeClass("mfp-" + a + "-holder");
            var r = t["get" + n.charAt(0).toUpperCase() + n.slice(1)](i, t.currTemplate[n]);
            t.appendContent(r, n), i.preloaded = !0, T(m, i), a = i.type, t.container.prepend(t.contentContainer), T("AfterChange")
        },
        appendContent: function(e, i) {
            t.content = e, e ? t.st.showCloseBtn && t.st.closeBtnInside && t.currTemplate[i] === !0 ? t.content.find(".mfp-close").length || t.content.append(S()) : t.content = e : t.content = "", T(u), t.container.addClass("mfp-" + i + "-holder"), t.contentContainer.append(t.content)
        },
        parseEl: function(i) {
            var n = t.items[i],
            o = n.type;
            if (n = n.tagName ? {
                el: e(n)
            } : {
                data: n,
                src: n.src
            }, n.el) {
                for (var a = t.types, r = 0; a.length > r; r++)
                    if (n.el.hasClass("mfp-" + a[r])) {
                        o = a[r];
                        break
                    }
                n.src = n.el.attr("data-mfp-src"), n.src || (n.src = n.el.attr("href"))
            }
            return n.type = o || t.st.type || "inline", n.index = i, n.parsed = !0, t.items[i] = n, T("ElementParse", n), t.items[i]
        },
        addGroup: function(e, i) {
            var n = function(n) {
                n.mfpEl = this, t._openClick(n, e, i)
            };
            i || (i = {});
            var o = "click.magnificPopup";
            i.mainEl = e, i.items ? (i.isObj = !0, e.off(o).on(o, n)) : (i.isObj = !1, i.delegate ? e.off(o).on(o, i.delegate, n) : (i.items = e, e.off(o).on(o, n)))
        },
        _openClick: function(i, n, o) {
            var a = void 0 !== o.midClick ? o.midClick : e.magnificPopup.defaults.midClick;
            if (a || 2 !== i.which && !i.ctrlKey && !i.metaKey) {
                var r = void 0 !== o.disableOn ? o.disableOn : e.magnificPopup.defaults.disableOn;
                if (r)
                    if (e.isFunction(r)) {
                        if (!r.call(t)) 
                            return !0
                    } else
                    if (r > I.width()) 
                        return !0;
                i.type && (i.preventDefault(), t.isOpen && i.stopPropagation()), o.el = e(i.mfpEl), o.delegate && (o.items = n.find(o.delegate)), t.open(o)
            }
        },
        updateStatus: function(e, n) {
            if (t.preloader) {
                i !== e && t.container.removeClass("mfp-s-" + i), n || "loading" !== e || (n = t.st.tLoading);
                var o = {
                    status: e,
                    text: n
                };
                T("UpdateStatus", o), e = o.status, n = o.text, t.preloader.html(n), t.preloader.find("a").on("click", function(e) {
                    e.stopImmediatePropagation()
                }), t.container.addClass("mfp-s-" + e), i = e
            }
        },
        _addClassToMFP: function(e) {
            t.bgOverlay.addClass(e), t.wrap.addClass(e)
        },
        _removeClassFromMFP: function(e) {
            this.bgOverlay.removeClass(e), t.wrap.removeClass(e)
        },
        _hasScrollBar: function(e) {
            return (t.isIE7 ? o.height() : document.body.scrollHeight) > (e || I.height())
        },
        _parseMarkup: function(t, i, n) {
            var o;
            n.data && (i = e.extend(n.data, i)), T(p, [t, i, n]), e.each(i, function(e, i) {
                if (void 0 === i || i === !1) 
                    return !0;
                if (o = e.split("_"), o.length > 1) {
                    var n = t.find(v + "-" + o[0]);
                    if (n.length > 0) {
                        var a = o[1];
                        "replaceWith" === a ? n[0] !== i[0] && n.replaceWith(i) : "img" === a ? n.is("img") ? n.attr("src", i) : n.replaceWith('<img src="' + i + '" class="' + n.attr("class") + '" />') : n.attr(o[1], i)
                    }
                } else 
                    t.find(v + "-" + e).html(i)
            })
        },
        _getScrollbarSize: function() {
            if (void 0 === t.scrollbarSize) {
                var e = document.createElement("div");
                e.id = "mfp-sbm", e.style.cssText = "width: 99px; height: 99px; overflow: scroll; position: absolute; top: -9999px;", document.body.appendChild(e), t.scrollbarSize = e.offsetWidth - e.clientWidth, document.body.removeChild(e)
            }
            return t.scrollbarSize
        }
    }, e.magnificPopup = {
        instance: null,
        proto: w.prototype,
        modules: [],
        open: function(e, t) {
            return P(), e || (e = {}), e.isObj = !0, e.index = t || 0, this.instance.open(e)
        },
        close: function() {
            return e.magnificPopup.instance.close()
        },
        registerModule: function(t, i) {
            i.options && (e.magnificPopup.defaults[t] = i.options), e.extend(this.proto, i.proto), this.modules.push(t)
        },
        defaults: {
            disableOn: 0,
            key: null,
            midClick: !1,
            mainClass: "",
            preloader: !0,
            focus: "",
            closeOnContentClick: !1,
            closeOnBgClick: !0,
            closeBtnInside: !0,
            showCloseBtn: !0,
            enableEscapeKey: !0,
            modal: !1,
            alignTop: !1,
            removalDelay: 0,
            fixedContentPos: "auto",
            fixedBgPos: "auto",
            overflowY: "auto",
            closeMarkup: '<button title="%title%" type="button" class="mfp-close">&times;</button>',
            tClose: "Close (Esc)",
            tLoading: "Loading..."
        }
    }, e.fn.magnificPopup = function(i) {
        P();
        var n = e(this);
        if ("string" == typeof i)
            if ("open" === i) {
                var o, a = b ? n.data("magnificPopup") : n[0].magnificPopup,
                r = parseInt(arguments[1], 10) || 0;
                a.items ? o = a.items[r] : (o = n, a.delegate && (o = o.find(a.delegate)), o = o.eq(r)), t._openClick({
                    mfpEl: o
                }, n, a)
            } else 
                t.isOpen && t[i].apply(t, Array.prototype.slice.call(arguments, 1));
        else 
            b ? n.data("magnificPopup", i) : n[0].magnificPopup = i, t.addGroup(n, i);
        return n
    };
    var z, M, B, H = "inline",
    L = function() {
        B && (M.after(B.addClass(z)).detach(), B = null)
    };
    e.magnificPopup.registerModule(H, {
        options: {
            hiddenClass: "hide",
            markup: "",
            tNotFound: "Content not found"
        },
        proto: {
            initInline: function() {
                t.types.push(H), x(l + "." + H, function() {
                    L()
                })
            },
            getInline: function(i, n) {
                if (L(), i.src) {
                    var o = t.st.inline,
                    a = e(i.src);
                    if (a.length) {
                        var r = a[0].parentNode;
                        r && r.tagName && (M || (z = o.hiddenClass, M = k(z), z = "mfp-" + z), B = a.after(M).detach().removeClass(z)), t.updateStatus("ready")
                    } else 
                        t.updateStatus("error", o.tNotFound), a = e("<div>");
                    return i.inlineElement = a, a
                }
                return t.updateStatus("ready"), t._parseMarkup(n, {}, i), n
            }
        }
    });
    var A, F = "ajax",
    j = function() {
        A && n.removeClass(A)
    };
    e.magnificPopup.registerModule(F, {
        options: {
            settings: null,
            cursor: "mfp-ajax-cur",
            tError: '<a href="%url%">The content</a> could not be loaded.'
        },
        proto: {
            initAjax: function() {
                t.types.push(F), A = t.st.ajax.cursor, x(l + "." + F, function() {
                    j(), t.req && t.req.abort()
                })
            },
            getAjax: function(i) {
                A && n.addClass(A), t.updateStatus("loading");
                var o = e.extend({
                    url: i.src,
                    success: function(n, o, a) {
                        var r = {
                            data: n,
                            xhr: a
                        };
                        T("ParseAjax", r), t.appendContent(e(r.data), F), i.finished = !0, j(), E(), setTimeout(function() {
                            t.wrap.addClass(h)
                        }, 16), t.updateStatus("ready"), T("AjaxContentAdded")
                    },
                    error: function() {
                        j(), i.finished = i.loadError = !0, t.updateStatus("error", t.st.ajax.tError.replace("%url%", i.src))
                    }
                }, t.st.ajax.settings);
                return t.req = e.ajax(o), ""
            }
        }
    });
    var N, W = function(i) {
        if (i.data && void 0 !== i.data.title) 
            return i.data.title;
        var n = t.st.image.titleSrc;
        if (n) {
            if (e.isFunction(n)) 
                return n.call(t, i);
            if (i.el) 
                return i.el.attr(n) || ""
        }
        return ""
    };
    e.magnificPopup.registerModule("image", {
        options: {
            markup: '<div class="mfp-figure"><div class="mfp-close"></div><div class="mfp-img"></div><div class="mfp-bottom-bar"><div class="mfp-title"></div><div class="mfp-counter"></div></div></div>',
            cursor: "mfp-zoom-out-cur",
            titleSrc: "title",
            verticalFit: !0,
            tError: '<a href="%url%">The image</a> could not be loaded.'
        },
        proto: {
            initImage: function() {
                var e = t.st.image,
                i = ".image";
                t.types.push("image"), x(f + i, function() {
                    "image" === t.currItem.type && e.cursor && n.addClass(e.cursor)
                }), x(l + i, function() {
                    e.cursor && n.removeClass(e.cursor), I.off("resize" + v)
                }), x("Resize" + i, t.resizeImage), t.isLowIE && x("AfterChange", t.resizeImage)
            },
            resizeImage: function() {
                var e = t.currItem;
                if (e.img && t.st.image.verticalFit) {
                    var i = 0;
                    t.isLowIE && (i = parseInt(e.img.css("padding-top"), 10) + parseInt(e.img.css("padding-bottom"), 10)), e.img.css("max-height", t.wH - i)
                }
            },
            _onImageHasSize: function(e) {
                e.img && (e.hasSize = !0, N && clearInterval(N), e.isCheckingImgSize = !1, T("ImageHasSize", e), e.imgHidden && (t.content && t.content.removeClass("mfp-loading"), e.imgHidden = !1))
            },
            findImageSize: function(e) {
                var i = 0,
                n = e.img[0],
                o = function(a) {
                    N && clearInterval(N), N = setInterval(function() {
                        return n.naturalWidth > 0 ? (t._onImageHasSize(e), void 0) : (i > 200 && clearInterval(N), i++, 3 === i ? o(10) : 40 === i ? o(50) : 100 === i && o(500), void 0)
                    }, a)
                };
                o(1)
            },
            getImage: function(i, n) {
                var o = 0,
                a = function() {
                    i && (i.img[0].complete ? (i.img.off(".mfploader"), i === t.currItem && (t._onImageHasSize(i), t.updateStatus("ready")), i.hasSize = !0, i.loaded = !0, T("ImageLoadComplete")) : (o++, 200 > o ? setTimeout(a, 100) : r()))
                }, r = function() {
                    i && (i.img.off(".mfploader"), i === t.currItem && (t._onImageHasSize(i), t.updateStatus("error", s.tError.replace("%url%", i.src))), i.hasSize = !0, i.loaded = !0, i.loadError = !0)
                }, s = t.st.image,
                l = n.find(".mfp-img");
                if (l.length) {
                    var c = new Image;
                    c.className = "mfp-img", i.img = e(c).on("load.mfploader", a).on("error.mfploader", r), c.src = i.src, l.is("img") && (i.img = i.img.clone()), i.img[0].naturalWidth > 0 && (i.hasSize = !0)
                }
                return t._parseMarkup(n, {
                    title: W(i),
                    img_replaceWith: i.img
                }, i), t.resizeImage(), i.hasSize ? (N && clearInterval(N), i.loadError ? (n.addClass("mfp-loading"), t.updateStatus("error", s.tError.replace("%url%", i.src))) : (n.removeClass("mfp-loading"), t.updateStatus("ready")), n) : (t.updateStatus("loading"), i.loading = !0, i.hasSize || (i.imgHidden = !0, n.addClass("mfp-loading"), t.findImageSize(i)), n)
            }
        }
    });
    var R, Z = function() {
        return void 0 === R && (R = void 0 !== document.createElement("p").style.MozTransform), R
    };
    e.magnificPopup.registerModule("zoom", {
        options: {
            enabled: !1,
            easing: "ease-in-out",
            duration: 300,
            opener: function(e) {
                return e.is("img") ? e : e.find("img")
            }
        },
        proto: {
            initZoom: function() {
                var e = t.st.zoom,
                i = ".zoom";
                if (e.enabled && t.supportsTransition) {
                    var n, o, a = e.duration,
                    r = function(t) {
                        var i = t.clone().removeAttr("style").removeAttr("class").addClass("mfp-animated-image"),
                        n = "all " + e.duration / 1e3 + "s " + e.easing,
                        o = {
                            position: "fixed",
                            zIndex: 9999,
                            left: 0,
                            top: 0,
                            "-webkit-backface-visibility": "hidden"
                        }, a = "transition";
                        return o["-webkit-" + a] = o["-moz-" + a] = o["-o-" + a] = o[a] = n, i.css(o), i
                    }, s = function() {
                        t.content.css("visibility", "visible")
                    };
                    x("BuildControls" + i, function() {
                        if (t._allowZoom()) {
                            if (clearTimeout(n), t.content.css("visibility", "hidden"), image = t._getItemToZoom(), !image) 
                                return s(), void 0;
                            o = r(image), o.css(t._getOffset()), t.wrap.append(o), n = setTimeout(function() {
                                o.css(t._getOffset(!0)), n = setTimeout(function() {
                                    s(), setTimeout(function() {
                                        o.remove(), image = o = null, T("ZoomAnimationEnded")
                                    }, 16)
                                }, a)
                            }, 16)
                        }
                    }), x(c + i, function() {
                        if (t._allowZoom()) {
                            if (clearTimeout(n), t.st.removalDelay = a, !image) {
                                if (image = t._getItemToZoom(), !image) 
                                    return;
                                o = r(image)
                            }
                            o.css(t._getOffset(!0)), t.wrap.append(o), t.content.css("visibility", "hidden"), setTimeout(function() {
                                o.css(t._getOffset())
                            }, 16)
                        }
                    }), x(l + i, function() {
                        t._allowZoom() && (s(), o && o.remove())
                    })
                }
            },
            _allowZoom: function() {
                return "image" === t.currItem.type
            },
            _getItemToZoom: function() {
                return t.currItem.hasSize ? t.currItem.img : !1
            },
            _getOffset: function(i) {
                var n;
                n = i ? t.currItem.img : t.st.zoom.opener(t.currItem.el || t.currItem);
                var o = n.offset(),
                a = parseInt(n.css("padding-top"), 10),
                r = parseInt(n.css("padding-bottom"), 10);
                o.top -= e(window).scrollTop() - a;
                var s = {
                    width: n.width(),
                    height: (b ? n.innerHeight() : n[0].offsetHeight) - r - a
                };
                return Z() ? s["-moz-transform"] = s.transform = "translate(" + o.left + "px," + o.top + "px)" : (s.left = o.left, s.top = o.top), s
            }
        }
    });
    var q = "iframe",
    D = "//about:blank",
    K = function(e) {
        if (t.currTemplate[q]) {
            var i = t.currTemplate[q].find("iframe");
            i.length && (e || (i[0].src = D), t.isIE8 && i.css("display", e ? "block" : "none"))
        }
    };
    e.magnificPopup.registerModule(q, {
        options: {
            markup: '<div class="mfp-iframe-scaler"><div class="mfp-close"></div><iframe class="mfp-iframe" src="//about:blank" frameborder="0" allowfullscreen></iframe></div>',
            srcAction: "iframe_src",
            patterns: {
                youtube: {
                    index: "youtube.com",
                    id: "v=",
                    src: "//www.youtube.com/embed/%id%?autoplay=1"
                },
                vimeo: {
                    index: "vimeo.com/",
                    id: "/",
                    src: "//player.vimeo.com/video/%id%?autoplay=1"
                },
                gmaps: {
                    index: "//maps.google.",
                    src: "%id%&output=embed"
                }
            }
        },
        proto: {
            initIframe: function() {
                t.types.push(q), x("BeforeChange", function(e, t, i) {
                    t !== i && (t === q ? K() : i === q && K(!0))
                }), x(l + "." + q, function() {
                    K()
                })
            },
            getIframe: function(i, n) {
                var o = i.src,
                a = t.st.iframe;
                e.each(a.patterns, function() {
                    return o.indexOf(this.index) > - 1 ? (this.id && (o = "string" == typeof this.id ? o.substr(o.lastIndexOf(this.id) + this.id.length, o.length) : this.id.call(this, o)), o = this.src.replace("%id%", o), !1) : void 0
                });
                var r = {};
                return a.srcAction && (r[a.srcAction] = o), t._parseMarkup(n, r, i), t.updateStatus("ready"), n
            }
        }
    });
    var Y = function(e) {
        var i = t.items.length;
        return e > i - 1 ? e - i : 0 > e ? i + e : e
    }, U = function(e, t, i) {
        return e.replace("%curr%", t + 1).replace("%total%", i)
    };
    e.magnificPopup.registerModule("gallery", {
        options: {
            enabled: !1,
            arrowMarkup: '<button title="%title%" type="button" class="mfp-arrow mfp-arrow-%dir%"></button>',
            preload: [0, 2],
            navigateByImgClick: !0,
            arrows: !0,
            tPrev: "Previous (Left arrow key)",
            tNext: "Next (Right arrow key)",
            tCounter: "%curr% of %total%"
        },
        proto: {
            initGallery: function() {
                var i = t.st.gallery,
                n = ".mfp-gallery",
                a = Boolean(e.fn.mfpFastClick);
                return t.direction = !0, i && i.enabled ? (r += " mfp-gallery", x(f + n, function() {
                    i.navigateByImgClick && t.wrap.on("click" + n, ".mfp-img", function() {
                        return t.items.length > 1 ? (t.next(), !1) : void 0
                    }), o.on("keydown" + n, function(e) {
                        37 === e.keyCode ? t.prev() : 39 === e.keyCode && t.next()
                    })
                }), x("UpdateStatus" + n, function(e, i) {
                    i.text && (i.text = U(i.text, t.currItem.index, t.items.length))
                }), x(p + n, function(e, n, o, a) {
                    var r = t.items.length;
                    o.counter = r > 1 ? U(i.tCounter, a.index, r) : ""
                }), x("BuildControls" + n, function() {
                    if (t.items.length > 1 && i.arrows && !t.arrowLeft) {
                        var n = i.arrowMarkup,
                        o = t.arrowLeft = e(n.replace("%title%", i.tPrev).replace("%dir%", "left")).addClass(y),
                        r = t.arrowRight = e(n.replace("%title%", i.tNext).replace("%dir%", "right")).addClass(y),
                        s = a ? "mfpFastClick" : "click";
                        o[s](function() {
                            t.prev()
                        }), r[s](function() {
                            t.next()
                        }), t.isIE7 && (k("b", o[0], !1, !0), k("a", o[0], !1, !0), k("b", r[0], !1, !0), k("a", r[0], !1, !0)), t.container.append(o.add(r))
                    }
                }), x(m + n, function() {
                    t._preloadTimeout && clearTimeout(t._preloadTimeout), t._preloadTimeout = setTimeout(function() {
                        t.preloadNearbyImages(), t._preloadTimeout = null
                    }, 16)
                }), x(l + n, function() {
                    o.off(n), t.wrap.off("click" + n), t.arrowLeft && a && t.arrowLeft.add(t.arrowRight).destroyMfpFastClick(), t.arrowRight = t.arrowLeft = null
                }), void 0) : !1
            },
            next: function() {
                t.direction = !0, t.index = Y(t.index + 1), t.updateItemHTML()
            },
            prev: function() {
                t.direction = !1, t.index = Y(t.index - 1), t.updateItemHTML()
            },
            goTo: function(e) {
                t.direction = e >= t.index, t.index = e, t.updateItemHTML()
            },
            preloadNearbyImages: function() {
                var e, i = t.st.gallery.preload,
                n = Math.min(i[0], t.items.length),
                o = Math.min(i[1], t.items.length);
                for (e = 1;
                (t.direction ? o : n) >= e; e++) 
                    t._preloadItem(t.index + e);
                for (e = 1;
                (t.direction ? n : o) >= e; e++) 
                    t._preloadItem(t.index - e)
            },
            _preloadItem: function(i) {
                if (i = Y(i), !t.items[i].preloaded) {
                    var n = t.items[i];
                    n.parsed || (n = t.parseEl(i)), T("LazyLoad", n), "image" === n.type && (n.img = e('<img class="mfp-img" />').on("load.mfploader", function() {
                        n.hasSize = !0
                    }).on("error.mfploader", function() {
                        n.hasSize = !0, n.loadError = !0, T("LazyLoadError", n)
                    }).attr("src", n.src)), n.preloaded = !0
                }
            }
        }
    });
    var G = "retina";
    e.magnificPopup.registerModule(G, {
        options: {
            replaceSrc: function(e) {
                return e.src.replace(/\.\w+$/, function(e) {
                    return "@2x" + e
                })
            },
            ratio: 1
        },
        proto: {
            initRetina: function() {
                if (window.devicePixelRatio > 1) {
                    var e = t.st.retina,
                    i = e.ratio;
                    i = isNaN(i) ? i() : i, i > 1 && (x("ImageHasSize." + G, function(e, t) {
                        t.img.css({
                            "max-width": t.img[0].naturalWidth / i,
                            width: "100%"
                        })
                    }), x("ElementParse." + G, function(t, n) {
                        n.src = e.replaceSrc(n, i)
                    }))
                }
            }
        }
    }),
    function() {
        var t = 1e3,
        i = "ontouchstart" in window,
        n = function() {
            I.off("touchmove" + a + " touchend" + a)
        }, o = "mfpFastClick",
        a = "." + o;
        e.fn.mfpFastClick = function(o) {
            return e(this).each(function() {
                var r, s = e(this);
                if (i) {
                    var l, c, d, u, p, f;
                    s.on("touchstart" + a, function(e) {
                        u = !1, f = 1, p = e.originalEvent ? e.originalEvent.touches[0] : e.touches[0], c = p.clientX, d = p.clientY, I.on("touchmove" + a, function(e) {
                            p = e.originalEvent ? e.originalEvent.touches : e.touches, f = p.length, p = p[0], (Math.abs(p.clientX - c) > 10 || Math.abs(p.clientY - d) > 10) && (u = !0, n())
                        }).on("touchend" + a, function(e) {
                            n(), u || f > 1 || (r = !0, e.preventDefault(), clearTimeout(l), l = setTimeout(function() {
                                r = !1
                            }, t), o())
                        })
                    })
                }
                s.on("click" + a, function() {
                    r || o()
                })
            })
        }, e.fn.destroyMfpFastClick = function() {
            e(this).off("touchstart" + a + " click" + a), i && I.off("touchmove" + a + " touchend" + a)
        }
    }()
})(window.jQuery || window.Zepto);


/**
 * Copyright (c) 2007-2012 Ariel Flesler - aflesler(at)gmail(dot)com | http://flesler.blogspot.com
 * Dual licensed under MIT and GPL.
 * @author Ariel Flesler
 * @version 1.4.3
 */
;
(function($) {
    var h = $.scrollTo = function(a, b, c) {
        $(window).scrollTo(a, b, c)
    };
    h.defaults = {
        axis: 'xy',
        duration: parseFloat($.fn.jquery) >= 1.3 ? 0 : 1,
        limit: true
    };
    h.window = function(a) {
        return $(window)._scrollable()
    };
    $.fn._scrollable = function() {
        return this.map(function() {
            var a = this,
            isWin = !a.nodeName || $.inArray(a.nodeName.toLowerCase(), ['iframe', '#document', 'html', 'body']) != - 1;
            if (!isWin) 
                return a;
            var b = (a.contentWindow || a).document || a.ownerDocument || a;
            return /webkit/i.test(navigator.userAgent) || b.compatMode == 'BackCompat' ? b.body : b.documentElement
        })
    };
    $.fn.scrollTo = function(e, f, g) {
        if (typeof f == 'object') {
            g = f;
            f = 0
        }
        if (typeof g == 'function') 
            g = {
                onAfter: g
            };
        if (e == 'max') 
            e = 9e9;
        g = $.extend({}, h.defaults, g);
        f = f || g.duration;
        g.queue = g.queue && g.axis.length > 1;
        if (g.queue) 
            f /= 2;
        g.offset = both(g.offset);
        g.over = both(g.over);
        return this._scrollable().each(function() {
            if (!e) 
                return;
            var d = this,
            $elem = $(d),
            targ = e,
            toff, attr = {}, win = $elem.is('html,body');
            switch (typeof targ) {
            case 'number':
            case 'string':
                if (/^([+-]=)?\d+(\.\d+)?(px|%)?$/.test(targ)) {
                    targ = both(targ);
                    break
                }
                targ = $(targ, this);
                if (!targ.length) 
                    return;
            case 'object':
                if (targ.is || targ.style) 
                    toff = (targ = $(targ)).offset()
            }
            $.each(g.axis.split(''), function(i, a) {
                var b = a == 'x' ? 'Left' : 'Top',
                pos = b.toLowerCase(),
                key = 'scroll' + b,
                old = d[key],
                max = h.max(d, a);
                if (toff) {
                    attr[key] = toff[pos] + (win ? 0 : old - $elem.offset()[pos]);
                    if (g.margin) {
                        attr[key] -= parseInt(targ.css('margin' + b)) || 0;
                        attr[key] -= parseInt(targ.css('border' + b + 'Width')) || 0
                    }
                    attr[key] += g.offset[pos] || 0;
                    if (g.over[pos]) 
                        attr[key] += targ[a == 'x' ? 'width' : 'height']() * g.over[pos]
                } else {
                    var c = targ[pos];
                    attr[key] = c.slice && c.slice( - 1) == '%' ? parseFloat(c) / 100 * max : c
                }
                if (g.limit && /^\d+$/.test(attr[key])) 
                    attr[key] = attr[key] <= 0 ? 0 : Math.min(attr[key], max);
                if (!i && g.queue) {
                    if (old != attr[key]) 
                        animate(g.onAfterFirst);
                    delete attr[key]
                }
            });
            animate(g.onAfter);

            function animate(a) {
                $elem.animate(attr, f, g.easing, a && function() {
                    a.call(this, e, g)
                })
            }
        }).end()
    };
    h.max = function(a, b) {
        var c = b == 'x' ? 'Width' : 'Height',
        scroll = 'scroll' + c;
        if (!$(a).is('html,body')) 
            return a[scroll] - $(a)[c.toLowerCase()]();
        var d = 'client' + c,
        html = a.ownerDocument.documentElement,
        body = a.ownerDocument.body;
        return Math.max(html[scroll], body[scroll]) - Math.min(html[d], body[d])
    };

    function both(a) {
        return typeof a == 'object' ? a : {
            top: a,
            left: a
        }
    }
})(jQuery);


/*
 * jQuery One Page Nav Plugin
 * http://github.com/davist11/jQuery-One-Page-Nav
 *
 * Copyright (c) 2010 Trevor Davis (http://trevordavis.net)
 * Dual licensed under the MIT and GPL licenses.
 * Uses the same license as jQuery, see:
 * http://jquery.org/license
 *
 * @version 0.6
 */
(function(e, t, n, r) {
    var i = function(r, i) {
        this.elem = r;
        this.$elem = e(r);
        this.options = i;
        this.metadata = this.$elem.data("plugin-options");
        this.$nav = this.$elem.find("a");
        this.$win = e(t);
        this.sections = {};
        this.didScroll = false;
        this.$doc = e(n);
        this.docHeight = this.$doc.height()
    };
    i.prototype = {
        defaults: {
            currentClass: "current",
            changeHash: false,
            easing: "swing",
            filter: "",
            scrollSpeed: 750,
            scrollOffset: 0,
            scrollThreshold: .5,
            begin: false,
            end: false,
            scrollChange: false
        },
        init: function() {
            var t = this;
            t.config = e.extend({}, t.defaults, t.options, t.metadata);
            if (t.config.filter !== "") {
                t.$nav = t.$nav.filter(t.config.filter)
            }
            t.$nav.on("click.onePageNav", e.proxy(t.handleClick, t));
            t.getPositions();
            t.bindInterval();
            t.$win.on("resize.onePageNav", e.proxy(t.getPositions, t));
            return this
        },
        adjustNav: function(e, t) {
            e.$elem.find("." + e.config.currentClass).removeClass(e.config.currentClass);
            t.addClass(e.config.currentClass)
        },
        bindInterval: function() {
            var e = this;
            var t;
            e.$win.on("scroll.onePageNav", function() {
                e.didScroll = true
            });
            e.t = setInterval(function() {
                t = e.$doc.height();
                if (e.didScroll) {
                    e.didScroll = false;
                    e.scrollChange()
                }
                if (t !== e.docHeight) {
                    e.docHeight = t;
                    e.getPositions()
                }
            }, 250)
        },
        getHash: function(e) {
            return e.attr("href").split("#")[1]
        },
        getPositions: function() {
            var t = this;
            var n;
            var r;
            var i;
            t.$nav.each(function() {
                n = t.getHash(e(this));
                i = e("#" + n);
                if (i.length) {
                    r = i.offset().top;
                    t.sections[n] = Math.round(r) - t.config.scrollOffset
                }
            })
        },
        getSection: function(e) {
            var t = null;
            var n = Math.round(this.$win.height() * this.config.scrollThreshold);
            for (var r in this.sections) {
                if (this.sections[r] - n < e) {
                    t = r
                }
            }
            return t
        },
        handleClick: function(n) {
            var r = this;
            var i = e(n.currentTarget);
            var s = i.parent();
            var o = "#" + r.getHash(i);
            if (!s.hasClass(r.config.currentClass)) {
                if (r.config.begin) {
                    r.config.begin()
                }
                r.adjustNav(r, s);
                r.unbindInterval();
                e.scrollTo(o, r.config.scrollSpeed, {
                    axis: "y",
                    easing: r.config.easing,
                    offset: {
                        top: - r.config.scrollOffset
                    },
                    onAfter: function() {
                        if (r.config.changeHash) {
                            t.location.hash = o
                        }
                        r.bindInterval();
                        if (r.config.end) {
                            r.config.end()
                        }
                    }
                })
            }
            n.preventDefault()
        },
        scrollChange: function() {
            var e = this.$win.scrollTop();
            var t = this.getSection(e);
            var n;
            if (t !== null) {
                n = this.$elem.find('a[href$="#' + t + '"]').parent();
                if (!n.hasClass(this.config.currentClass)) {
                    this.adjustNav(this, n);
                    if (this.config.scrollChange) {
                        this.config.scrollChange(n)
                    }
                }
            }
        },
        unbindInterval: function() {
            clearInterval(this.t);
            this.$win.unbind("scroll.onePageNav")
        }
    };
    i.defaults = i.prototype.defaults;
    e.fn.onePageNav = function(e) {
        return this.each(function() {
            (new i(this, e)).init()
        })
    }
})(jQuery, window, document);

/*
 * jQuery FlexSlider v2.1
 * http://www.woothemes.com/flexslider/
 *
 * Copyright 2012 WooThemes
 * Free to use under the GPLv2 license.
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 * Contributing author: Tyler Smith (@mbmufffin)
 */

(function(d) {
    d.flexslider = function(j, l) {
        var a = d(j),
        c = d.extend({}, d.flexslider.defaults, l),
        e = c.namespace,
        q = "ontouchstart" in window || window.DocumentTouch && document instanceof DocumentTouch,
        u = q ? "touchend" : "click",
        m = "vertical" === c.direction,
        n = c.reverse,
        h = 0 < c.itemWidth,
        s = "fade" === c.animation,
        t = "" !== c.asNavFor,
        f = {};
        d.data(j, "flexslider", a);
        f = {
            init: function() {
                a.animating = !1;
                a.currentSlide = c.startAt;
                a.animatingTo = a.currentSlide;
                a.atEnd = 0 === a.currentSlide || a.currentSlide === a.last;
                a.containerSelector = c.selector.substr(0,
                c.selector.search(" "));
                a.slides = d(c.selector, a);
                a.container = d(a.containerSelector, a);
                a.count = a.slides.length;
                a.syncExists = 0 < d(c.sync).length;
                "slide" === c.animation && (c.animation = "swing");
                a.prop = m ? "top" : "marginLeft";
                a.args = {};
                a.manualPause = !1;
                var b = a,
                g;
                if (g = !c.video)
                    if (g = !s)
                        if (g = c.useCSS) 
                            a: {
                                g = document.createElement("div");
                                var p = ["perspectiveProperty", "WebkitPerspective", "MozPerspective", "OPerspective", "msPerspective"],
                                e;
                                for (e in p)
                                    if (void 0 !== g.style[p[e]]) {
                                        a.pfx = p[e].replace("Perspective", "").toLowerCase();
                                        a.prop = "-" + a.pfx + "-transform";
                                        g = !0;
                                        break a
                                    }
                                    g = !1
                                }
                b.transitions = g;
                "" !== c.controlsContainer && (a.controlsContainer = 0 < d(c.controlsContainer).length && d(c.controlsContainer));
                "" !== c.manualControls && (a.manualControls = 0 < d(c.manualControls).length && d(c.manualControls));
                c.randomize && (a.slides.sort(function() {
                    return Math.round(Math.random()) - 0.5
                }), a.container.empty().append(a.slides));
                a.doMath();
                t && f.asNav.setup();
                a.setup("init");
                c.controlNav && f.controlNav.setup();
                c.directionNav && f.directionNav.setup();
                c.keyboard &&
                (1 === d(a.containerSelector).length || c.multipleKeyboard) && d(document).bind("keyup", function(b) {
                    b = b.keyCode;
                    if (!a.animating && (39 === b || 37 === b)) 
                        b = 39 === b ? a.getTarget("next") : 37 === b ? a.getTarget("prev") : !1, a.flexAnimate(b, c.pauseOnAction)
                });
                c.mousewheel && a.bind("mousewheel", function(b, g) {
                    b.preventDefault();
                    var d = 0 > g ? a.getTarget("next") : a.getTarget("prev");
                    a.flexAnimate(d, c.pauseOnAction)
                });
                c.pausePlay && f.pausePlay.setup();
                c.slideshow && (c.pauseOnHover && a.hover(function() {
                    !a.manualPlay && !a.manualPause && a.pause()
                },
                function() {
                    !a.manualPause && !a.manualPlay && a.play()
                }), 0 < c.initDelay ? setTimeout(a.play, c.initDelay) : a.play());
                q && c.touch && f.touch();
                (!s || s && c.smoothHeight) && d(window).bind("resize focus", f.resize);
                setTimeout(function() {
                    c.start(a)
                }, 200)
            },
            asNav: {
                setup: function() {
                    a.asNav = !0;
                    a.animatingTo = Math.floor(a.currentSlide / a.move);
                    a.currentItem = a.currentSlide;
                    a.slides.removeClass(e + "active-slide").eq(a.currentItem).addClass(e + "active-slide");
                    a.slides.click(function(b) {
                        b.preventDefault();
                        b = d(this);
                        var g = b.index();
                        !d(c.asNavFor).data("flexslider").animating && !b.hasClass("active") && (a.direction = a.currentItem < g ? "next" : "prev", a.flexAnimate(g, c.pauseOnAction, !1, !0, !0))
                    })
                }
            },
            controlNav: {
                setup: function() {
                    a.manualControls ? f.controlNav.setupManual() : f.controlNav.setupPaging()
                },
                setupPaging: function() {
                    var b = 1,
                    g;
                    a.controlNavScaffold = d('<ol class="' + e + "control-nav " + e + ("thumbnails" === c.controlNav ? "control-thumbs" : "control-paging") + '"></ol>');
                    if (1 < a.pagingCount)
                        for (var p = 0; p < a.pagingCount; p++) 
                            g = "thumbnails" === c.controlNav ?
                            '<img src="' + a.slides.eq(p).attr("data-thumb") + '"/>' : "<a>" + b + "</a>", a.controlNavScaffold.append("<li>" + g + "</li>"), b++;
                    a.controlsContainer ? d(a.controlsContainer).append(a.controlNavScaffold) : a.append(a.controlNavScaffold);
                    f.controlNav.set();
                    f.controlNav.active();
                    a.controlNavScaffold.delegate("a, img", u, function(b) {
                        b.preventDefault();
                        b = d(this);
                        var g = a.controlNav.index(b);
                        b.hasClass(e + "active") || (a.direction = g > a.currentSlide ? "next" : "prev", a.flexAnimate(g, c.pauseOnAction))
                    });
                    q && a.controlNavScaffold.delegate("a",
                    "click touchstart", function(a) {
                        a.preventDefault()
                    })
                },
                setupManual: function() {
                    a.controlNav = a.manualControls;
                    f.controlNav.active();
                    a.controlNav.live(u, function(b) {
                        b.preventDefault();
                        b = d(this);
                        var g = a.controlNav.index(b);
                        b.hasClass(e + "active") || (g > a.currentSlide ? a.direction = "next" : a.direction = "prev", a.flexAnimate(g, c.pauseOnAction))
                    });
                    q && a.controlNav.live("click touchstart", function(a) {
                        a.preventDefault()
                    })
                },
                set: function() {
                    a.controlNav = d("." + e + "control-nav li " + ("thumbnails" === c.controlNav ? "img" : "a"),
                    a.controlsContainer ? a.controlsContainer : a)
                },
                active: function() {
                    a.controlNav.removeClass(e + "active").eq(a.animatingTo).addClass(e + "active")
                },
                update: function(b, c) {
                    1 < a.pagingCount && "add" === b ? a.controlNavScaffold.append(d("<li><a>" + a.count + "</a></li>")) : 1 === a.pagingCount ? a.controlNavScaffold.find("li").remove() : a.controlNav.eq(c).closest("li").remove();
                    f.controlNav.set();
                    1 < a.pagingCount && a.pagingCount !== a.controlNav.length ? a.update(c, b) : f.controlNav.active()
                }
            },
            directionNav: {
                setup: function() {
                    var b = d('<ul class="' +
                    e + 'direction-nav"><li><a class="' + e + 'prev" href="#">' + c.prevText + '</a></li><li><a class="' + e + 'next" href="#">' + c.nextText + "</a></li></ul>");
                    a.controlsContainer ? (d(a.controlsContainer).append(b), a.directionNav = d("." + e + "direction-nav li a", a.controlsContainer)) : (a.append(b), a.directionNav = d("." + e + "direction-nav li a", a));
                    f.directionNav.update();
                    a.directionNav.bind(u, function(b) {
                        b.preventDefault();
                        b = d(this).hasClass(e + "next") ? a.getTarget("next") : a.getTarget("prev");
                        a.flexAnimate(b, c.pauseOnAction)
                    });
                    q && a.directionNav.bind("click touchstart", function(a) {
                        a.preventDefault()
                    })
                },
                update: function() {
                    var b = e + "disabled";
                    1 === a.pagingCount ? a.directionNav.addClass(b) : c.animationLoop ? a.directionNav.removeClass(b) : 0 === a.animatingTo ? a.directionNav.removeClass(b).filter("." + e + "prev").addClass(b) : a.animatingTo === a.last ? a.directionNav.removeClass(b).filter("." + e + "next").addClass(b) : a.directionNav.removeClass(b)
                }
            },
            pausePlay: {
                setup: function() {
                    var b = d('<div class="' + e + 'pauseplay"><a></a></div>');
                    a.controlsContainer ?
                    (a.controlsContainer.append(b), a.pausePlay = d("." + e + "pauseplay a", a.controlsContainer)) : (a.append(b), a.pausePlay = d("." + e + "pauseplay a", a));
                    f.pausePlay.update(c.slideshow ? e + "pause" : e + "play");
                    a.pausePlay.bind(u, function(b) {
                        b.preventDefault();
                        d(this).hasClass(e + "pause") ? (a.manualPause = !0, a.manualPlay = !1, a.pause()) : (a.manualPause = !1, a.manualPlay = !0, a.play())
                    });
                    q && a.pausePlay.bind("click touchstart", function(a) {
                        a.preventDefault()
                    })
                },
                update: function(b) {
                    "play" === b ? a.pausePlay.removeClass(e + "pause").addClass(e +
                    "play").text(c.playText) : a.pausePlay.removeClass(e + "play").addClass(e + "pause").text(c.pauseText)
                }
            },
            touch: function() {
                function b(b) {
                    k = m ? d - b.touches[0].pageY : d - b.touches[0].pageX;
                    q = m ? Math.abs(k) < Math.abs(b.touches[0].pageX - e) : Math.abs(k) < Math.abs(b.touches[0].pageY - e);
                    if (!q || 500 < Number(new Date) - l) 
                        b.preventDefault(), !s && a.transitions && (c.animationLoop || (k /= 0 === a.currentSlide && 0 > k || a.currentSlide === a.last && 0 < k ? Math.abs(k) / r + 2 : 1), a.setProps(f + k, "setTouch"))
                }

                function g() {
                    j.removeEventListener("touchmove",
                    b, !1);
                    if (a.animatingTo === a.currentSlide && !q && null !== k) {
                        var h = n ? - k : k,
                        m = 0 < h ? a.getTarget("next") : a.getTarget("prev");
                        a.canAdvance(m) && (550 > Number(new Date) - l && 50 < Math.abs(h) || Math.abs(h) > r / 2) ? a.flexAnimate(m, c.pauseOnAction) : s || a.flexAnimate(a.currentSlide, c.pauseOnAction, !0)
                    }
                    j.removeEventListener("touchend", g, !1);
                    f = k = e = d = null
                }
                var d, e, f, r, k, l, q = !1;
                j.addEventListener("touchstart", function(k) {
                    a.animating ? k.preventDefault() : 1 === k.touches.length && (a.pause(), r = m ? a.h : a.w, l = Number(new Date), f = h && n && a.animatingTo ===
                    a.last ? 0 : h && n ? a.limit - (a.itemW + c.itemMargin) * a.move * a.animatingTo : h && a.currentSlide === a.last ? a.limit : h ? (a.itemW + c.itemMargin) * a.move * a.currentSlide : n ? (a.last - a.currentSlide + a.cloneOffset) * r : (a.currentSlide + a.cloneOffset) * r, d = m ? k.touches[0].pageY : k.touches[0].pageX, e = m ? k.touches[0].pageX : k.touches[0].pageY, j.addEventListener("touchmove", b, !1), j.addEventListener("touchend", g, !1))
                }, !1)
            },
            resize: function() {
                !a.animating && a.is(":visible") && (h || a.doMath(), s ? f.smoothHeight() : h ? (a.slides.width(a.computedW),
                a.update(a.pagingCount), a.setProps()) : m ? (a.viewport.height(a.h), a.setProps(a.h, "setTotal")) : (c.smoothHeight && f.smoothHeight(), a.newSlides.width(a.computedW), a.setProps(a.computedW, "setTotal")))
            },
            smoothHeight: function(b) {
                if (!m || s) {
                    var c = s ? a : a.viewport;
                    b ? c.animate({
                        height: a.slides.eq(a.animatingTo).height()
                    }, b) : c.height(a.slides.eq(a.animatingTo).height())
                }
            },
            sync: function(b) {
                var g = d(c.sync).data("flexslider"),
                e = a.animatingTo;
                switch (b) {
                case "animate":
                    g.flexAnimate(e, c.pauseOnAction, !1, !0);
                    break;
                case "play":
                    !g.playing && !g.asNav && g.play();
                    break;
                case "pause":
                    g.pause()
                }
            }
        };
        a.flexAnimate = function(b, g, p, j, l) {
            t && 1 === a.pagingCount && (a.direction = a.currentItem < b ? "next" : "prev");
            if (!a.animating && (a.canAdvance(b, l) || p) && a.is(":visible")) {
                if (t && j)
                    if (p = d(c.asNavFor).data("flexslider"), a.atEnd = 0 === b || b === a.count - 1, p.flexAnimate(b, !0, !1, !0, l), a.direction = a.currentItem < b ? "next" : "prev", p.direction = a.direction, Math.ceil((b + 1) / a.visible) - 1 !== a.currentSlide && 0 !== b) 
                        a.currentItem = b, a.slides.removeClass(e + "active-slide").eq(b).addClass(e +
                        "active-slide"), b = Math.floor(b / a.visible);
                    else 
                        return a.currentItem = b, a.slides.removeClass(e + "active-slide").eq(b).addClass(e + "active-slide"), !1;
                a.animating = !0;
                a.animatingTo = b;
                c.before(a);
                g && a.pause();
                a.syncExists && !l && f.sync("animate");
                c.controlNav && f.controlNav.active();
                h || a.slides.removeClass(e + "active-slide").eq(b).addClass(e + "active-slide");
                a.atEnd = 0 === b || b === a.last;
                c.directionNav && f.directionNav.update();
                b === a.last && (c.end(a), c.animationLoop || a.pause());
                if (s) 
                    q ? (a.slides.eq(a.currentSlide).css({
                        opacity: 0,
                        zIndex: 1
                    }), a.slides.eq(b).css({
                        opacity: 1,
                        zIndex: 2
                    }), a.slides.unbind("webkitTransitionEnd transitionend"), a.slides.eq(a.currentSlide).bind("webkitTransitionEnd transitionend", function() {
                        c.after(a)
                    }), a.animating = !1, a.currentSlide = a.animatingTo) : (a.slides.eq(a.currentSlide).fadeOut(c.animationSpeed, c.easing), a.slides.eq(b).fadeIn(c.animationSpeed, c.easing, a.wrapup));
                else {
                    var r = m ? a.slides.filter(":first").height() : a.computedW;
                    h ? (b = c.itemWidth > a.w ? 2 * c.itemMargin : c.itemMargin, b = (a.itemW + b) * a.move * a.animatingTo,
                    b = b > a.limit && 1 !== a.visible ? a.limit : b) : b = 0 === a.currentSlide && b === a.count - 1 && c.animationLoop && "next" !== a.direction ? n ? (a.count + a.cloneOffset) * r : 0 : a.currentSlide === a.last && 0 === b && c.animationLoop && "prev" !== a.direction ? n ? 0 : (a.count + 1) * r : n ? (a.count - 1 - b + a.cloneOffset) * r : (b + a.cloneOffset) * r;
                    a.setProps(b, "", c.animationSpeed);
                    if (a.transitions) {
                        if (!c.animationLoop || !a.atEnd) 
                            a.animating = !1, a.currentSlide = a.animatingTo;
                        a.container.unbind("webkitTransitionEnd transitionend");
                        a.container.bind("webkitTransitionEnd transitionend",
                        function() {
                            a.wrapup(r)
                        })
                    } else 
                        a.container.animate(a.args, c.animationSpeed, c.easing, function() {
                            a.wrapup(r)
                        })
                    }
                c.smoothHeight && f.smoothHeight(c.animationSpeed)
            }
        };
        a.wrapup = function(b) {
            !s && !h && (0 === a.currentSlide && a.animatingTo === a.last && c.animationLoop ? a.setProps(b, "jumpEnd") : a.currentSlide === a.last && (0 === a.animatingTo && c.animationLoop) && a.setProps(b, "jumpStart"));
            a.animating = !1;
            a.currentSlide = a.animatingTo;
            c.after(a)
        };
        a.animateSlides = function() {
            a.animating || a.flexAnimate(a.getTarget("next"))
        };
        a.pause =
        function() {
            clearInterval(a.animatedSlides);
            a.playing = !1;
            c.pausePlay && f.pausePlay.update("play");
            a.syncExists && f.sync("pause")
        };
        a.play = function() {
            a.animatedSlides = setInterval(a.animateSlides, c.slideshowSpeed);
            a.playing = !0;
            c.pausePlay && f.pausePlay.update("pause");
            a.syncExists && f.sync("play")
        };
        a.canAdvance = function(b, g) {
            var d = t ? a.pagingCount - 1 : a.last;
            return g ? !0 : t && a.currentItem === a.count - 1 && 0 === b && "prev" === a.direction ? !0 : t && 0 === a.currentItem && b === a.pagingCount - 1 && "next" !== a.direction ? !1 : b === a.currentSlide && !t ? !1 : c.animationLoop ? !0 : a.atEnd && 0 === a.currentSlide && b === d && "next" !== a.direction ? !1 : a.atEnd && a.currentSlide === d && 0 === b && "next" === a.direction ? !1 : !0
        };
        a.getTarget = function(b) {
            a.direction = b;
            return "next" === b ? a.currentSlide === a.last ? 0 : a.currentSlide + 1 : 0 === a.currentSlide ? a.last : a.currentSlide - 1
        };
        a.setProps = function(b, g, d) {
            var e, f = b ? b : (a.itemW + c.itemMargin) * a.move * a.animatingTo;
            e = - 1 * function() {
                if (h) 
                    return "setTouch" === g ? b : n && a.animatingTo === a.last ? 0 : n ? a.limit - (a.itemW + c.itemMargin) * a.move * a.animatingTo : a.animatingTo ===
                    a.last ? a.limit : f;
                switch (g) {
                case "setTotal":
                    return n ? (a.count - 1 - a.currentSlide + a.cloneOffset) * b : (a.currentSlide + a.cloneOffset) * b;
                case "setTouch":
                    return b;
                case "jumpEnd":
                    return n ? b : a.count * b;
                case "jumpStart":
                    return n ? a.count * b : b;
                default:
                    return b
                }
            }() + "px";
            a.transitions && (e = m ? "translate3d(0," + e + ",0)" : "translate3d(" + e + ",0,0)", d = void 0 !== d ? d / 1E3 + "s" : "0s", a.container.css("-" + a.pfx + "-transition-duration", d));
            a.args[a.prop] = e;
            (a.transitions || void 0 === d) && a.container.css(a.args)
        };
        a.setup = function(b) {
            if (s) 
                a.slides.css({
                    width: "100%",
                    "float": "left",
                    marginRight: "-100%",
                    position: "relative"
                }), "init" === b && (q ? a.slides.css({
                    opacity: 0,
                    display: "block",
                    webkitTransition: "opacity " + c.animationSpeed / 1E3 + "s ease",
                    zIndex: 1
                }).eq(a.currentSlide).css({
                    opacity: 1,
                    zIndex: 2
                }) : a.slides.eq(a.currentSlide).fadeIn(c.animationSpeed, c.easing)), c.smoothHeight && f.smoothHeight();
            else {
                var g, p;
                "init" === b && (a.viewport = d('<div class="' + e + 'viewport"></div>').css({
                    overflow: "hidden",
                    position: "relative"
                }).appendTo(a).append(a.container), a.cloneCount = 0, a.cloneOffset =
                0, n && (p = d.makeArray(a.slides).reverse(), a.slides = d(p), a.container.empty().append(a.slides)));
                c.animationLoop && !h && (a.cloneCount = 2, a.cloneOffset = 1, "init" !== b && a.container.find(".clone").remove(), a.container.append(a.slides.first().clone().addClass("clone")).prepend(a.slides.last().clone().addClass("clone")));
                a.newSlides = d(c.selector, a);
                g = n ? a.count - 1 - a.currentSlide + a.cloneOffset : a.currentSlide + a.cloneOffset;
                m && !h ? (a.container.height(200 * (a.count + a.cloneCount) + "%").css("position", "absolute").width("100%"),
                setTimeout(function() {
                    a.newSlides.css({
                        display: "block"
                    });
                    a.doMath();
                    a.viewport.height(a.h);
                    a.setProps(g * a.h, "init")
                }, "init" === b ? 100 : 0)) : (a.container.width(200 * (a.count + a.cloneCount) + "%"), a.setProps(g * a.computedW, "init"), setTimeout(function() {
                    a.doMath();
                    a.newSlides.css({
                        width: a.computedW,
                        "float": "left",
                        display: "block"
                    });
                    c.smoothHeight && f.smoothHeight()
                }, "init" === b ? 100 : 0))
            }
            h || a.slides.removeClass(e + "active-slide").eq(a.currentSlide).addClass(e + "active-slide")
        };
        a.doMath = function() {
            var b = a.slides.first(),
            d = c.itemMargin,
            e = c.minItems,
            f = c.maxItems;
            a.w = a.width();
            a.h = b.height();
            a.boxPadding = b.outerWidth() - b.width();
            h ? (a.itemT = c.itemWidth + d, a.minW = e ? e * a.itemT : a.w, a.maxW = f ? f * a.itemT : a.w, a.itemW = a.minW > a.w ? (a.w - d * e) / e : a.maxW < a.w ? (a.w - d * f) / f : c.itemWidth > a.w ? a.w : c.itemWidth, a.visible = Math.floor(a.w / (a.itemW + d)), a.move = 0 < c.move && c.move < a.visible ? c.move : a.visible, a.pagingCount = Math.ceil((a.count - a.visible) / a.move + 1), a.last = a.pagingCount - 1, a.limit = 1 === a.pagingCount ? 0 : c.itemWidth > a.w ? (a.itemW + 2 * d) * a.count - a.w -
            d : (a.itemW + d) * a.count - a.w - d) : (a.itemW = a.w, a.pagingCount = a.count, a.last = a.count - 1);
            a.computedW = a.itemW - a.boxPadding
        };
        a.update = function(b, d) {
            a.doMath();
            h || (b < a.currentSlide ? a.currentSlide += 1 : b <= a.currentSlide && 0 !== b && (a.currentSlide -= 1), a.animatingTo = a.currentSlide);
            if (c.controlNav && !a.manualControls)
                if ("add" === d && !h || a.pagingCount > a.controlNav.length) 
                    f.controlNav.update("add");
                else
                if ("remove" === d && !h || a.pagingCount < a.controlNav.length) 
                    h && a.currentSlide > a.last && (a.currentSlide -= 1, a.animatingTo -= 1),
                    f.controlNav.update("remove", a.last);
            c.directionNav && f.directionNav.update()
        };
        a.addSlide = function(b, e) {
            var f = d(b);
            a.count += 1;
            a.last = a.count - 1;
            m && n ? void 0 !== e ? a.slides.eq(a.count - e).after(f) : a.container.prepend(f) : void 0 !== e ? a.slides.eq(e).before(f) : a.container.append(f);
            a.update(e, "add");
            a.slides = d(c.selector + ":not(.clone)", a);
            a.setup();
            c.added(a)
        };
        a.removeSlide = function(b) {
            var e = isNaN(b) ? a.slides.index(d(b)) : b;
            a.count -= 1;
            a.last = a.count - 1;
            isNaN(b) ? d(b, a.slides).remove() : m && n ? a.slides.eq(a.last).remove() :
            a.slides.eq(b).remove();
            a.doMath();
            a.update(e, "remove");
            a.slides = d(c.selector + ":not(.clone)", a);
            a.setup();
            c.removed(a)
        };
        f.init()
    };
    d.flexslider.defaults = {
        namespace: "flex-",
        selector: ".slides > li",
        animation: "fade",
        easing: "swing",
        direction: "horizontal",
        reverse: !1,
        animationLoop: !0,
        smoothHeight: !1,
        startAt: 0,
        slideshow: !0,
        slideshowSpeed: 7E3,
        animationSpeed: 600,
        initDelay: 0,
        randomize: !1,
        pauseOnAction: !0,
        pauseOnHover: !1,
        useCSS: !0,
        touch: !0,
        video: !1,
        controlNav: !0,
        directionNav: !0,
        prevText: "Previous",
        nextText: "Next",
        keyboard: !0,
        multipleKeyboard: !1,
        mousewheel: !1,
        pausePlay: !1,
        pauseText: "Pause",
        playText: "Play",
        controlsContainer: "",
        manualControls: "",
        sync: "",
        asNavFor: "",
        itemWidth: 0,
        itemMargin: 0,
        minItems: 0,
        maxItems: 0,
        move: 0,
        start: function() {},
        before: function() {},
        after: function() {},
        end: function() {},
        added: function() {},
        removed: function() {}
    };
    d.fn.flexslider = function(j) {
        void 0 === j && (j = {});
        if ("object" === typeof j) 
            return this.each(function() {
                var a = d(this),
                c = a.find(j.selector ? j.selector : ".slides > li");
                1 === c.length ? (c.fadeIn(400),
                j.start && j.start(a)) : void 0 == a.data("flexslider") && new d.flexslider(this, j)
            });
        var l = d(this).data("flexslider");
        switch (j) {
        case "play":
            l.play();
            break;
        case "pause":
            l.pause();
            break;
        case "next":
            l.flexAnimate(l.getTarget("next"), !0);
            break;
        case "prev":
        case "previous":
            l.flexAnimate(l.getTarget("prev"), !0);
            break;
        default:
            "number" === typeof j && l.flexAnimate(j, !0)
        }
    }
})(jQuery);


/*global jQuery */
/*jshint multistr:true browser:true */
/*!
 * FitVids 1.0
 *
 * Copyright 2011, Chris Coyier - http://css-tricks.com + Dave Rupert - http://daverupert.com
 * Credit to Thierry Koblentz - http://www.alistapart.com/articles/creating-intrinsic-ratios-for-video/
 * Released under the WTFPL license - http://sam.zoy.org/wtfpl/
 *
 * Date: Thu Sept 01 18:00:00 2011 -0500
 */
(function(e) {
    "use strict";
    e.fn.fitVids = function(t) {
        var n = {
            customSelector: null
        };
        if (!document.getElementById("fit-vids-style")) {
            var r = document.createElement("div"),
            i = document.getElementsByTagName("base")[0] || document.getElementsByTagName("script")[0];
            r.className = "fit-vids-style";
            r.id = "fit-vids-style";
            r.style.display = "none";
            r.innerHTML = "­<style>                 .fluid-width-video-wrapper {                   width: 100%;                                position: relative;                         padding: 0;                              }                                                                                       .fluid-width-video-wrapper iframe,          .fluid-width-video-wrapper object,          .fluid-width-video-wrapper embed {             position: absolute;                         top: 0;                                     left: 0;                                    width: 100%;                                height: 100%;                            }                                         </style>";
            i.parentNode.insertBefore(r, i)
        }
        if (t) {
            e.extend(n, t)
        }
        return this.each(function() {
            var t = ["iframe[src*='player.vimeo.com']", "iframe[src*='youtube.com']", "iframe[src*='youtube-nocookie.com']", "iframe[src*='kickstarter.com'][src*='video.html']", "object", "embed"];
            if (n.customSelector) {
                t.push(n.customSelector)
            }
            var r = e(this).find(t.join(","));
            r = r.not("object object");
            r.each(function() {
                var t = e(this);
                if (this.tagName.toLowerCase() === "embed" && t.parent("object").length || t.parent(".fluid-width-video-wrapper").length) {
                    return
                }
                var n = this.tagName.toLowerCase() === "object" || t.attr("height") && !isNaN(parseInt(t.attr("height"), 10)) ? parseInt(t.attr("height"), 10) : t.height(),
                r = !isNaN(parseInt(t.attr("width"), 10)) ? parseInt(t.attr("width"), 10) : t.width(),
                i = n / r;
                if (!t.attr("id")) {
                    var s = "fitvid" + Math.floor(Math.random() * 999999);
                    t.attr("id", s)
                }
                t.wrap('<div class="fluid-width-video-wrapper"></div>').parent(".fluid-width-video-wrapper").css("padding-top", i * 100 + "%");
                t.removeAttr("height").removeAttr("width")
            })
        })
    }
})(jQuery);
