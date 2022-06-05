(function($) {
    if (!$.PRESTO) {
        $.PRESTO = new Object()
    };
    $.PRESTO.Carousel = function(g, h, j, k) {
        var l = this;
        l.$el = $(g);
        l.el = g;
        l.$el.data("PRESTO.Carousel", l);
        l.init = function() {
            if (typeof(h) == "undefined" || h === null)
                h = 400;
            if (typeof(j) == "undefined" || j === null)
                j = 300;
            l.width = h;
            l.height = j;
            l.options = $.extend({}, $.PRESTO.Carousel.defaultOptions, k);
            l.$el.addClass('j-carousel' + l.options.animation);
            l.activeSeat =- 1;
            l.seats = l.$el.find('.seat');
            l.seatCount = l.seats.length;
            l.bindControls();
            l.loadMedia();
            if (l.seatCount > 0) {
                l.run()
            }
        };
        l.loadMedia = function() {
            l.$el.children('.struct').children('.image').each(function(i) {
                var src = $(this).html();
                setTimeout(function() {
                    var a = new Image();
                    $(a).load(function() {
                        $(this).hide();
                        $(l.seats[i]).children('.media-box').find('img').replaceWith(a);
                        $(this).show();
                        $(l.seats[i]).addClass('ready').trigger("imageLoaded");
                    }).css({
                        'width': l.width,
                        'height': l.height
                    }).attr('src', src);
                }, (i * 500));
            });
        };
        l.run = function(i) {
            if (l.$el.is(':visible') || l.activeSeat==-1) {
                if (!l.$el.hasClass('loading')) {
                    l.$el.addClass('loading');
                    l.stop();
                    l.lastActiveSeat = l.activeSeat;
                    if (i >= 0) {
                        l.activeSeat = i
                    } else if (i < 0) {
                        l.activeSeat--
                    } else {
                        l.activeSeat++
                    }
                    if (l.activeSeat > l.seatCount - 1) {
                        l.activeSeat = 0
                    } else if (l.activeSeat < 0) {
                        l.activeSeat = l.seatCount - 1
                    }
                    l.$el.trigger('beforeSlide', [l.activeSeat, l.lastActiveSeat]);
                    if ($(l.seats[l.activeSeat]).hasClass('ready')) {
                        l.renderActive()
                    } else {
                        if (l.$el.children('.loader').length == 0) {
                            l.$el.append('<img class="loader" src="/info/images/ajax_loader.gif" alt="" style="position:absolute;right:10px;top:10px;z-index:9999" />');
                        } else {
                            l.$el.children('.loader').show();
                        }
                        $(l.seats[l.activeSeat]).on('imageLoaded', function() {
                            l.$el.children('.loader').hide();
                            l.renderActive();
                        });
                    }
                }
            } else {
                l.carouselTimer = setTimeout(function() {
                    l.run()
                }, l.options.interval)
            }
        };
        l.stop = function() {
            clearTimeout(l.carouselTimer)
        };
        l.play = function() {
            l.setupRerun()
        };
        l.setupRerun = function() {
            l.$el.removeClass('loading');
            l.$el.trigger('afterSlide', [l.activeSeat, l.lastActiveSeat]);
            if (l.options.auto) {
                l.carouselTimer = setTimeout(function() {
                    l.run()
                }, l.options.interval)
            }
        };
        l.renderActive = function() {
            if (l.lastActiveSeat >= 0) {
                $(l.seats[l.lastActiveSeat]).removeClass('active');
                l.$el.find(l.options.control).eq(l.lastActiveSeat).removeClass('active')
            }
            $(l.seats[l.activeSeat]).addClass('active');
            l.$el.find(l.options.control).eq(l.activeSeat).addClass('active');
            switch (l.options.animation) {
            case 0:
                l.layout0();
                break;
            case 1:
                l.layout1();
                break;
            case 2:
                l.layout2();
                break;
            case 3:
                l.layout3();
                break;
            case 4:
                l.layout4();
                break
            }
        };
        l.layout0 = function() {
            if (l.lastActiveSeat >= 0) {
                $(l.seats[l.lastActiveSeat]).hide()
            }
            $(l.seats[l.activeSeat]).show();
            l.setupRerun()
        };
        l.layout1 = function() {
            if (l.lastActiveSeat >= 0) {
                l.seats.css({
                    'z-index': 1
                });
                $(l.seats[l.lastActiveSeat]).css({
                    'z-index': 3
                });
                $(l.seats[l.activeSeat]).css({
                    'z-index': 2
                }).show();
                $(l.seats[l.lastActiveSeat]).fadeOut(l.options.animationSpeed, function() {
                    l.setupRerun()
                })
            } else {
                $(l.seats[l.activeSeat]).show();
                l.setupRerun()
            }
        };
        l.layout2 = function() {
            if (l.lastActiveSeat >= 0) {
                var c = $(l.seats[l.lastActiveSeat]).children('.story-box'), d =- (c.outerHeight());
                c.animate({
                    'bottom': d
                }, l.options.animationSpeed, 'easeInCubic', function() {
                    $(l.seats[l.lastActiveSeat]).hide();
                    var a = $(l.seats[l.activeSeat]).show().children('.story-box');
                    var b =- (a.outerHeight());
                    a.css({
                        'bottom': b
                    }).animate({
                        'bottom': 0
                    }, l.options.animationSpeed, 'easeOutCubic', function() {
                        l.setupRerun()
                    })
                })
            } else {
                $(l.seats[l.activeSeat]).show();
                l.setupRerun()
            }
        };
        l.layout3 = function(s) {
            if (l.lastActiveSeat >= 0) {
                var c = l.$el.children('.seats'), d = (l.activeSeat * $(l.seats[l.activeSeat]).width()) * ( - 1), o;
                if (l.lastActiveSeat == l.seatCount - 1 && l.activeSeat == 0) {
                    var cl = $(l.seats[l.activeSeat]).clone().appendTo(c);
                    d = ((l.lastActiveSeat + 1) * cl.width()) * ( - 1);
                    o = function() {
                        c.css('left', 0);
                        cl.remove()
                    }
                } else if (l.lastActiveSeat == 0 && l.activeSeat == l.seatCount - 1) {
                    var cl = $(l.seats[l.activeSeat]).clone().prependTo(c);
                    c.css('left', cl.width() * ( - 1));
                    d = 0;
                    o = function() {
                        c.css('left', (l.activeSeat * $(l.seats[l.activeSeat]).width()) * ( - 1));
                        cl.remove()
                    }
                }
                s = (typeof(s) == "undefined") ? l.options.animationSpeed : s;
                c.animate({
                    'left': d
                }, s, 'easeInCubic', function() {
                    if (typeof(o) != "undefined") {
                        o()
                    }
                    l.setupRerun()
                })
            } else {
                l.$el.children('.seats').swipe({
                    swipe: function(e, dir) {
                        if (dir === "left") {
                            l.run()
                        } else if (dir === "right") {
                            l.run( - 1)
                        }
                    },
                    excludedElements: '',
                    fallbackToMouseEvents: false,
                    allowPageScroll: 'vertical'
                });
                l.setupRerun()
            }
        };
        l.layout4 = function() {
            if (l.collapsedWidth === undefined) {
                l.collapsedWidth = (l.$el.width() - l.width) / (l.seatCount - 1);
                l.seats.each(function(i) {
                    var a = (i > 1) ? l.width + (l.collapsedWidth * (i - 1)): i * l.width;
                    var b = l.collapsedWidth * i;
                    $(this).data('offset', {
                        right: a,
                        left: b
                    });
                    $(this).css({
                        'left': a,
                        'z-index': i * 10
                    }).append('<div class="mask" style="width:' + (l.width - 1) + 'px;height:' + l.height + 'px"><div class="label"></div></div>')
                });
                $(l.seats[l.activeSeat]).children('.media-box, .story-box').show().end().children('.mask').addClass('strip')
            }
            if (!l.$el.hasClass('initialized')) {
                if (l.activeSeat == (l.seatCount - 1)) {
                    l.activeSeat = 0;
                    l.seats.not(l.seats.eq(l.activeSeat)).removeClass('active');
                    l.$el.find(l.options.control).removeClass('active').eq(l.activeSeat).addClass('active');
                    l.$el.addClass('initialized');
                    l.seats.children('.media-box').show();
                    l.setupRerun()
                } else {
                    l.activeSeat++;
                    l.loadMedia(l.activeSeat)
                }
            } else {
                $(l.seats[l.activeSeat]).children('.mask').addClass('strip');
                $(l.seats[l.lastActiveSeat]).children('.story-box').hide().end().children('.mask').removeClass('strip');
                if (l.activeSeat > l.lastActiveSeat) {
                    var c = l.seats.slice(l.lastActiveSeat + 1, l.activeSeat + 1);
                    var d = "easeInBack"
                } else {
                    var c = l.seats.slice(l.activeSeat + 1);
                    var d = "easeOutExpo"
                }
                c.each(function(i) {
                    var a = l.activeSeat > l.lastActiveSeat ? $(this).data('offset').left: $(this).data('offset').right;
                    $(this).animate({
                        'left': a
                    }, l.options.animationSpeed, d, function() {
                        if (i == (c.length - 1)) {
                            $(l.seats[l.activeSeat]).children('.story-box').fadeIn('slow', function() {
                                l.setupRerun()
                            })
                        }
                    })
                })
            }
        };
        l.bindControls = function() {
            l.$el.on('click', function(e) {
                if ($(e.target).is(l.options.control) || $(e.target).parent().is(l.options.control)) {
                    e.preventDefault();
                    var i =- 1;
                    var a = l.$el.find(l.options.control);
                    if ($(e.target).parent().is(l.options.control)) {
                        i = a.index($(e.target).parent())
                    } else {
                        i = a.index($(e.target))
                    }
                    if (i >= 0 && i != l.activeSeat) {
                        l.run(i)
                    }
                }
            }).hover(function() {
                $(this).addClass('hovered')
            }, function() {
                $(this).removeClass('hovered')
            }).children(l.options.next).click(function(e) {
                e.preventDefault();
                l.run()
            }).end().children(l.options.prev).click(function(e) {
                e.preventDefault();
                l.run( - 1)
            }).end().children(l.options.playPause).html(function() {
                if (l.options.auto) {
                    $(this).addClass('pause');
                    return 'Pause'
                } else {
                    $(this).addClass('play');
                    return 'Play'
                }
            }).click(function(e) {
                e.preventDefault();
                if ($(this).hasClass('pause')) {
                    l.stop();
                    l.options.auto = false;
                    $(this).html('Play').removeClass('pause').addClass('play')
                } else {
                    $(this).html('Pause').removeClass('play').addClass('pause');
                    l.options.auto = true;
                    l.setupRerun()
                }
            })
        };
        l.init()
    };
    $.PRESTO.Carousel.defaultOptions = {
        auto: true,
        interval: 5000,
        animationSpeed: 400,
        animation: 1,
        control: '.control',
        next: '.next',
        prev: '.prev',
        playPause: '.pp'
    };
    $.fn.presto_Carousel = function(a, b, c) {
        return this.each(function() {
            (new $.PRESTO.Carousel(this, a, b, c))
        })
    };
    $.fn.getPRESTO_Carousel = function() {
        return this.data('PRESTO.Carousel')
    }
})(jQuery);
