(function($) {
    $.fn.easySlider = function(options) {
        var defaults = {
            prevId: 'prevBtn',
            prevText: 'Previous',
            nextId: 'nextBtn',
            nextText: 'Next',
            pauseText: 'Pause',
            playText: 'Play',
            playPauseId: 'playPauseBtn',
            controlsShow: true,
            controlsBefore: '',
            controlsAfter: '',
            controlsFade: true,
            firstId: 'firstBtn',
            firstText: 'First',
            firstShow: false,
            lastId: 'lastBtn',
            lastText: 'Last',
            lastShow: false,
            vertical: false,
            speed: 800,
            auto: false,
            pause: 2000,
            continuous: false,
            numeric: false,
            numericId: 'controls',
            summaryShow: false,
            summaryId: 'sliderSummary'
        };
        var options = $.extend(defaults, options);
        this.each(function() {
            var obj = $(this);
            var s = $("li", obj).length;
            var w = $("li", obj).width();
            var h = $("li", obj).height();
            var clickable = true;
            obj.width(w);
            obj.height(h);
            obj.css("overflow", "hidden");
            var ts = s - 1;
            var t = 0, q = 0;
            if (!options.vertical) {
                $("ul", obj).css('width', s * w);
            } else {
                $("ul", obj).css('height', s * h);
            }
            if (!options.vertical)
                $("li", obj).css('float', 'left');
            if (options.continuous) {
                if (!options.vertical) {
                    $("ul", obj).prepend($("ul li:last-child", obj).clone().css("margin-left", "-" + w + "px"));
                    $("ul", obj).append($("ul li:nth-child(2)", obj).clone());
                    $("ul", obj).css('width', (s + 1) * w);
                } else {
                    $("ul", obj).prepend($("ul li:last-child", obj).clone()).css("margin-top", "-" + h + "px");
                    $("ul", obj).append($("ul li:nth-child(2)", obj).clone());
                    $("ul", obj).css('height', (s + 1) * h);
                    t = 1;
                    ts = s;
                    q = 1;
                };
            };
            if (options.controlsShow) {
                var html = options.controlsBefore;
                if (options.numeric) {
                    html += '<ol id="' + options.numericId + '"></ol>';
                } else {
                    if (options.firstShow)
                        html += '<span id="' + options.firstId + '"><a href=\"javascript:void(0);\">' + options.firstText + '</a></span>';
                    html += ' <span id="' + options.prevId + '"><a href=\"javascript:void(0);\">' + options.prevText + '</a></span>';
                    if (options.auto) {
                        html += ' <span id="' + options.playPauseId + '"><a href=\"javascript:void(0);\" class=\"pause\">' + options.pauseText + '</a></span>';
                    }
                    html += ' <span id="' + options.nextId + '"><a href=\"javascript:void(0);\">' + options.nextText + '</a></span>';
                    if (options.lastShow)
                        html += ' <span id="' + options.lastId + '"><a href=\"javascript:void(0);\">' + options.lastText + '</a></span>';
                };
                html += options.controlsAfter;
                $(obj).after(html);
            };
            if (options.summaryShow) {
                $(obj).after('<span id="' + options.summaryId + '"></span>');
            };
            if (options.numeric) {
                for (var i = 0; i < s; i++) {
                    $(document.createElement("li")).attr('id', options.numericId + (i + 1)).html('<a rel=' + i + ' href=\"javascript:void(0);\">' + (i + 1) + '</a>').appendTo($("#" + options.numericId)).click(function() {
                        animate($("a", $(this)).attr('rel'));
                    });
                };
            } else {
                $("a", "#" + options.nextId).click(function() {
                    animate("next");
                });
                $("a", "#" + options.prevId).click(function() {
                    animate("prev");
                });
                $("a", "#" + options.firstId).click(function() {
                    animate("first");
                });
                $("a", "#" + options.lastId).click(function() {
                    animate("last");
                });
                if (options.auto) {
                    $("a", "#" + options.playPauseId).click(function() {
                        if (timeout) {
                            clearTimeout(timeout);
                            timeout = 0;
                            $("a", "#" + options.playPauseId).text(options.playText).attr('class', 'play');
                        } else {
                            timeout = setTimeout(function() {
                                animate("next");
                            }, options.pause);
                            $("a", "#" + options.playPauseId).text(options.pauseText).attr('class', 'pause');
                        }
                    });
                }
            };
            function setCurrent(i) {
                i = parseInt(i) + 1;
                $("li", "#" + options.numericId).removeClass("current");
                $("li#" + options.numericId + i).addClass("current");
            };
            function showSummary(i) {
                i = parseInt(i) + 1;
                $("#" + options.summaryId).text(i + '/' + s);
            }
            function adjust() {
                if (t > ts)
                    t = q;
                if (t < q)
                    t = ts;
                if (!options.vertical) {
                    $("ul", obj).css("margin-left", (t * w*-1));
                } else {
                    $("ul", obj).css("margin-top", (t * h*-1));
                }
                clickable = true;
                if (options.numeric)
                    setCurrent(t);
                if (options.summaryShow)
                    showSummary(t);
            };
            function animate(dir) {
                if (clickable) {
                    clickable = false;
                    if (options.auto && timeout) {
                        clearTimeout(timeout);
                        timeout = 0;
                    }
                    var ot = t;
                    switch (dir) {
                    case"next":
                        t = (ot >= ts) ? (options.continuous ? t + 1 : ts) : t + 1;
                        break;
                    case"prev":
                        t = (t <= 0) ? (options.continuous ? t - 1 : 0) : t - 1;
                        break;
                    case"first":
                        t = 0;
                        break;
                    case"last":
                        t = ts;
                        break;
                    default:
                        t = dir;
                        break;
                    };
                    var diff = Math.abs(ot - t);
                    var speed = diff * options.speed;
                    if (!options.vertical) {
                        p = (t * w*-1);
                        $("ul", obj).animate({
                            marginLeft: p
                        }, {
                            queue: false,
                            duration: speed,
                            complete: adjust
                        });
                    } else {
                        p = (t * h*-1);
                        $("ul", obj).animate({
                            marginTop: p
                        }, {
                            queue: false,
                            duration: speed,
                            complete: adjust
                        });
                    };
                    if (!options.continuous && options.controlsFade) {
                        if (t == ts) {
                            $("a", "#" + options.nextId).hide();
                            $("a", "#" + options.lastId).hide();
                        } else {
                            $("a", "#" + options.nextId).show();
                            $("a", "#" + options.lastId).show();
                        };
                        if (t == 0) {
                            $("a", "#" + options.prevId).hide();
                            $("a", "#" + options.firstId).hide();
                        } else {
                            $("a", "#" + options.prevId).show();
                            $("a", "#" + options.firstId).show();
                        };
                    };
                    if (options.auto && $("a", "#" + options.playPauseId).hasClass('pause')) {
                        timeout = setTimeout(function() {
                            animate("next");
                        }, diff * options.speed + options.pause);
                    };
                };
            };
            var timeout;
            if (options.auto) {
                ;
                timeout = setTimeout(function() {
                    animate("next");
                }, options.pause);
            };
            if (options.numeric)
                setCurrent(0);
            if (options.summaryShow)
                showSummary(0);
            if (!options.continuous && options.controlsFade) {
                $("a", "#" + options.prevId).hide();
                $("a", "#" + options.firstId).hide();
            };
        });
    };
})(jQuery);

