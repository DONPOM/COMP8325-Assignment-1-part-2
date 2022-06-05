$(function() {
    (function(C, B, z, A) {
        if (C && B) {
            C.on("mouseover", function() {
                C.addClass("ipthover")
            }).on("mouseout", function() {
                C.removeClass("ipthover")
            });
            B.on("focus", function() {
                C.addClass("iptfocus")
            }).on("blur", function() {
                C.removeClass("iptfocus")
            }).on("render", function(F) {
                var D = C.parent().find(".bdsug");
                var E = D.find("li").length;
                if (E >= 5) {
                    D.addClass("bdsugbg")
                } else {
                    D.removeClass("bdsugbg")
                }
            })
        }
        if (z && A) {
            z.on("mouseover", function() {
                A.addClass("btnhover")
            }).on("mouseout", function() {
                A.removeClass("btnhover");
                A.removeClass("s_btn_h")
            }).on("mousedown", function() {
                A.removeClass("btnhover");
                A.addClass("s_btn_h")
            }).on("mouseup", function() {
                A.addClass("btnhover");
                A.removeClass("s_btn_h")
            })
        }
    })($(".s_ipt_wr"), $(".s_ipt"), $(".s_btn_wr"), $(".s_btn"));
    var n = $("#wrapper"), o = $("#u"), s = $("#u .pf,#u1 .pf"), b;
    var q = 0, c = 0, m = 0, u, x, l, i;
    var f = 70;
    var j = $("<input type='hidden' name='rsv_enter' value='1'>");
    $("#form").append(j);
    $("#su").on("mousedown", function() {
        j.val(0)
    });
    function v(z) {
        var B = new RegExp("^\\s+|\\s+\x24"), A = $("#kw").val().replace(B, "");
        ns_c({
            fm: "behs",
            tab: z,
            query: encodeURIComponent(A),
            un: encodeURIComponent(bds.comm.user || "")
        })
    }
    $(document).on("click", function() {
        p()
    });
    function w(B, C) {
        var A = true;
        var z;
        B.mouseover(function() {
            C.show();
            if (z) {
                clearTimeout(z);
                z = false
            }
        });
        B.mouseout(function() {
            if (z) {
                clearTimeout(z);
                z = false
            }
            z = setTimeout(function() {
                C.hide()
            }, 500)
        });
        C.mouseover(function() {
            if (z) {
                clearTimeout(z);
                z = false
            }
        });
        C.mouseout(function() {
            if (z) {
                clearTimeout(z);
                z = false
            }
            z = setTimeout(function() {
                C.hide()
            }, 500)
        })
    }
    o.delegate(".username", "mouseover", function() {
        if ($(this).nextAll(".usermenu").length == 0) {
            l = $('<div class="usermenu"><a href="http://i.baidu.com" onmousedown="return user_c({\'fm\':\'set\',\'tab\':\'uc_center\'})">个人中心</a><a href="http://passport.baidu.com" name="tj_user">帐号设置</a><a href="http://passport.baidu.com/?logout&tpl=mn&u=" class="logout" name="tj_logout">退出</a></div>').insertAfter(this);
            l.delegate(".logout", "click", function() {
                location.href = this.href + encodeURIComponent(location.href);
                return false
            });
            i = $('<div class="bdnuarrow arrowusermenu"></div>').insertAfter(this);
            l.click(function(A) {
                A.stopPropagation()
            });
            i.click(function(A) {
                A.stopPropagation()
            });
            w($(this), l.add(i))
        }
        var z = $(this).offset();
        l.offset({
            top: z.top + 30,
            left: z.left - 15
        });
        i.offset({
            top: z.top + 18,
            left: z.left
        })
    });
    s.on("click", function(z) {
        return false
    });
    s.on("mouseover", function(z) {
        c=!!x;
        b = $(this);
        o = $(this).parent();
        z.stopPropagation();
        z.preventDefault();
        h();
        if (!c) {
            g();
            c = 1;
            x.show().hover(function() {
                k()
            }, function() {
                d()
            });
            u.show().hover(function() {
                k()
            }, function() {
                d()
            })
        }
        k()
    }).on("mouseout", function() {
        d()
    });
    var e = 200;
    var y, t;
    function r() {
        y && clearTimeout(y);
        t && clearTimeout(t)
    }
    function k() {
        var z = b.offset();
        x && x.css({
            left: z.left - 20,
            top: z.top + 30
        });
        u && u.css({
            left: z.left,
            top: z.top + 18
        });
        x && x.show();
        u && u.show();
        l && l.hide();
        i && i.hide();
        b.addClass("pfhover");
        r(y)
    }
    function d() {
        y = setTimeout(function() {
            x && x.hide();
            u && u.hide();
            b && b.removeClass("pfhover")
        }, e)
    }
    function p() {
        u && u.hide();
        i && i.hide();
        x && x.hide();
        l && l.hide()
    }
    function h() {
        if (!c) {
            u = $("<div>", {
                "class": "bdnuarrow bdpfarrow"
            });
            u.appendTo(n)
        }
    }
    function a(E) {
        var D = "<style>#surveywrapper{width:540px;height:350px;background:#fff;position:fixed;margin:180px 0 0 -270px;font-size:13px;z-index:491;padding:18px;top:0;left:50%;}#surveywrapper form{margin:15px 0 0 42px;}#surveywrapper .c-icon-close{position:absolute;top:12px;right:18px;cursor:pointer;}#surveywrapper .title{font-size:16px;height:30px;}#surveywrapper .row,#surveywrapper h5{height:30px;}#surveywrapper h5{margin:5px 0 0 0;height:30px;font-size:13px;}#surveywrapper iframe{height:0;width:0;padding:0;margin:0;border:none;}#surveywrapper textarea{width:420px;height:60px}#surveywrapper .s_btn{background-image: none;width: 100px; height: 34px; color: white; letter-spacing: 1px; background: #3385ff; border-bottom: 1px solid #2d78f4; outline: medium;}#surveywrapper .btn{margin-top:13px;}</style>";
        var A = D + '<div id="surveywrapper"><h3>用户调查</h3><i class="c-icon c-icon-close"></i><form target="survey_iframe" method="post" action="http://f3.baidu.com/index.php/feedback/zx/bdis"><div class="title">您关闭即输即得的理由是什么？</div><div class="row"><label><input type="radio" name="reason" value="1" checked />太难用了，不习惯</label></div><div class="row"><label><input type="radio" name="reason" value="2" />预测老出错，打扰我搜索</label></div><div class="row"><label><input type="radio" name="reason" value="3" />速度太慢了，半天没响应</label></div><div class="row"><label><input type="radio" name="reason" value="4" />页面有错误，根本没法搜</label></div><h5>其他理由</h5><textarea name="text"></textarea><div class="btn"><input type="submit" value="提交" class="s_btn"></div><input type="hidden" name="baiduid" value="' + escapeHTML(Cookie.get("BAIDUID")) + '"/><input type="hidden" name="url" value="' + escapeHTML(location.href) + '"/><input type="hidden" name="useragent" value="' + escapeHTML(navigator.userAgent) + '"/></form><iframe width="100" height="30" frameborder="no" border="0" marginwidth="0" marginheight="0" scrolling="no" allowtransparency="yes" name="survey_iframe"></iframe></div>';
        var I, H = false;
        var B = $("body");
        function F() {
            if (!I) {
                I = $("<div id='_mask'/>").css({
                    opacity: 0.3,
                    position: "absolute",
                    background: "#000",
                    "z-index": 490,
                    top: 0,
                    left: 0
                })
            }
            if (!H) {
                H = true;
                var J = $(window);
                I.width(Math.max(J.width(), B.width()));
                I.height(Math.max(J.height(), B.height()));
                I.appendTo(B)
            }
        }
        function G() {
            if (I && H) {
                H = false;
                I.remove()
            }
        }
        var C;
        function z() {
            F();
            C = $(A).appendTo(B);
            C.delegate(".c-icon-close", "click", function() {
                G();
                C.remove();
                return false
            });
            C.delegate("form", "submit", function() {
                C.find("iframe").on("load", function() {
                    E()
                });
                setTimeout(E, 1000)
            })
        }
        z()
    }
    function g() {
        var B = $("<a class='setpref' href='javascript:;'>搜索设置</a>").on("mousedown", function() {
            v("tj_setting");
            return false
        }).on("click", function() {
            var D = $("<div>", {
                id: "prefpanel"
            });
            if (!m) {
                D.appendTo(n);
                $.ajax({
                    dataType: "script",
                    cache: true,
                    url: "http://s1.bdstatic.com/r/www/cache/static/home/js/pf_instant_search_f54d2b4c.js"
                });
                m = 1
            }
            D.show().animate({
                top: "0px",
                opacity: 0.98
            }, f * 5);
            return false
        });
        var C = $("<a href='javascript:;'>意见反馈</a>").on("mousedown", function() {
            v("tj_feedback");
            return false
        }).on("click", function() {
            $.getScript("http://s1.bdstatic.com/r/www/cache/static/plugins/feedback_d057de2a.js", function() {
                if (bds.comm.ishome) {
                    var E = {
                        product_id: 24,
                        entrance_id: 0,
                        needGuide: false,
                        showPosition: "center"
                    }
                } else {
                    var E = {
                        product_id: 18,
                        entrance_id: 0,
                        needGuide: true,
                        showPosition: "right"
                    }
                }
                var D = {
                    username: bds.comm.username,
                    query: bds.comm.query,
                    fb_qid: bds.comm.qid
                };
                bds.qa.ShortCut.initRightBar(E, D)
            });
            return false
        });
        var z = $("<a href='//www.baidu.com/gaoji/advanced.html'>高级搜索</a>").on("mousedown", function() {
            v("tj_advsearch");
            return false
        }).on("click", function() {
            if ($("#kw").val()) {
                location.href = this.href + "?q=" + encodeURIComponent($("#kw").val());
                return false
            }
        });
        if (bds.comm.supportis || Cookie.get("ORIGIN") == 2) {
            var A;
            if (UPS.get("isSwitch") == 0) {
                A = $("<a href='javascript:;'>开启预测</a>").click(function() {
                    UPS.set("isSwitch", 1);
                    UPS.save(function() {
                        location.reload()
                    });
                    return false
                })
            } else {
                A = $("<a href='javascript:;'>关闭预测</a>").click(function() {
                    UPS.set("isSwitch", 0);
                    UPS.save(function() {
                        location.reload()
                    });
                    return false
                })
            }
        }
        x = $("<div>", {
            "class": "bdpfmenu"
        }).append(B).append(z);
        if (A) {
            x.append(A);
            x.append("<a href=\"http://www.baidu.com/search/is/index.html\" target=\"_blank\"  onmousedown=\"return ns_c({'fm':'tab','tab':'jsjd'})\">了解新版</a>")
        }
        x.append(C);
        x.appendTo(n);
        x.on("click", function(D) {
            D.stopPropagation();
            b && b.removeClass("pfhover");
            u.hide();
            x.hide()
        })
    }
});
$(function() {
    var n = $("#wrapper"), o = $("#u1"), b = $("#u1 .bri");
    var q = 0, c = 0, l = 0, u = 0, f, w, s, v;
    var i, m;
    var g = 70;
    function x(z) {
        var B = new RegExp("^\\s+|\\s+\x24"), A = $("#kw").get(0).value.replace(B, "");
        ns_c({
            fm: "behs",
            tab: z,
            query: encodeURIComponent(A),
            un: encodeURIComponent(bds.comm.user || "")
        })
    }
    $(document).on("click", function() {
        p()
    });
    $("#kw").on("click", function() {
        p()
    });
    b.on("click", function(z) {
        z.stopPropagation();
        z.preventDefault();
        return false
    });
    b.on("mouseover", function(z) {
        z.stopPropagation();
        z.preventDefault();
        a();
        if (!u) {
            k();
            u = 1;
            v.hover(function() {}, function() {
                d()
            });
            s.hover(function() {}, function() {
                d()
            })
        }
        j()
    }).on("mouseout", function() {});
    var e = 100;
    var y, t;
    $(window).on("index_off", function() {
        v && v.hide();
        s && s.hide()
    });
    function r() {
        y && clearTimeout(y);
        t && clearTimeout(t)
    }
    function j() {
        var z = $(".briguide");
        z && z.hide();
        if (v) {
            v.css({
                display: "block",
                opacity: "0",
                "min-height": m
            });
            if ("undefined" == typeof(document.body.style.maxHeight)) {
                v.css({
                    height: m
                })
            }
            v.find(".briscrollwrapper").scrollTop(0);
            v.css({
                display: "none",
                opacity: "1"
            }).fadeIn(e)
        }
        b.fadeOut(e);
        w && w.hide();
        s && s.show();
        f && f.render($(window).height() - b.offset().top - 34 - 20);
        r(t)
    }
    function d() {
        t = setTimeout(function() {
            v && v.fadeOut(e);
            s && s.hide();
            b.fadeIn(e)
        }, e)
    }
    function p() {
        w && w.hide();
        s && s.hide();
        v && v.hide()
    }
    function h() {
        if (!c) {
            w = $("<div>", {
                "class": "bdnuarrow bdpfarrow"
            });
            w.appendTo(o)
        }
    }
    function a() {
        if (!u) {
            s = $("<div>", {
                "class": "bdnuarrow bdbriarrow"
            });
            s.appendTo(o)
        }
    }
    function k() {
        v = $("<div>", {
            "class": "bdbri"
        }).appendTo($(".head_wrapper"));
        v.on("click", function(G) {
            G.stopPropagation()
        });
        if (o.hasClass("bdbrilink")) {} else {
            v.addClass("bdbriimg").html("<div class='bdmainlink'><div class='bdbriimgtitle'>更多产品</div><div class='briscrollwrapperContainer'><div class='briscrollwrapper'><div class='bdbriwrapper'><a href='http://www.nuomi.com/?cid=002540' name='tj_nuomi'><span class='bdbriimgitem_1'></span>糯米</a><a href='http://zhidao.baidu.com' name='tj_zhidao'><span class='bdbriimgitem_2'></span>知道</a><a href='http://music.baidu.com' name='tj_mp3'><span class='bdbriimgitem_3'></span>音乐</a><a href='http://image.baidu.com' name='tj_img'><span class='bdbriimgitem_4'></span>图片</a><a href='http://wenku.baidu.com' name='tj_wenku'><span class='bdbriimgitem_5'></span>文库</a><a href='http://top.baidu.com' name='tj_bang'><span class='bdbriimgitem_6'></span>风云榜</a><a href='http://e.baidu.com/?refer=888' name='tj_tuiguang'><span class='bdbriimgitem_7'></span>百度推广</a><div class='bdbrievenmore'><a href='//www.baidu.com/more/' name='tj_more'>全部产品&gt;&gt;</a></div></div></div></div></div>");
            var E = v.find(".bdothlink");
            var F = v.find(".bdbrievenmore");
            var z = v.find(".briscrollwrapper");
            var B = 1;
            function D() {
                z.animate({
                    scrollTop: v.height()
                }, g * 5, "swing", function() {
                    B = 0;
                    v.find(".bdbriscroll-ctrl-scroll").fadeIn(g)
                })
            }
            function C(ac) {
                var V = this;
                this.options = ac;
                var aD = ac.scrollbar || $("<div>").get(0), O = ac.content, ar = $(ac.content).children().get(0), N = ac.initPos || 0, aq = ac.initDom || null, az = ac.mousewheel || true, Q = ac.mousewheellock || false, al = ac.wheeldelta || 1, ae = ac.ctrlblock || 0, an = ac.step || 0.1, W = ac.length, am = ac.scale || 0, ak = ac.theme || "", aI = ac.refresh || false;
                var ax = 0, ay = 0, M = 0, aA = function(aL) {
                    var aK = parseInt(ax - ay);
                    if (aK > 0) {
                        var aL = aL.value;
                        O.scrollTop = aK * aL
                    }
                }, ab = $("<div>", {
                    "class": "bdbriscroll-up"
                }).get(0), J = $("<div>", {
                    "class": "bdbriscroll-down"
                }).get(0), aa = $("<div>", {
                    "class": "bdbriscroll-axis"
                }).get(0), L = $("<div>", {
                    "class": "bdbriscroll-slider OP_LOG_BTN"
                }).get(0), Z = $("<div>", {
                    "class": "bdbriscroll-s-top"
                }).get(0), aj = $("<div>", {
                    "class": "bdbriscroll-s-bottom"
                }).get(0), aJ = $("<div>", {
                    "class": "bdbriscroll-s-block"
                }).get(0), aB = 0, av = ae || 0, P = 0, aw = av, R = 0, ag = 0, ap = 0, I = 0, Y = null, G = null, aG, au, ah;
                var ad = function() {
                    aC = false;
                    H = false
                };
                if (!ac.scrollbar) {
                    $(O).after($(aD))
                }
                $(O).addClass("bdbriscroll-ctrl-content");
                $(aD).addClass("bdbriscroll-ctrl-scroll");
                $(aD).attr("data-click", '{fm:"beha"}');
                this.render = function(aL) {
                    if (!aI) {
                        clearInterval(ah)
                    }
                    try {
                        ay = O.offsetHeight;
                        M = aD.offsetHeight;
                        ax = ar.offsetHeight
                    } catch (aM) {}
                    aB = aL || W || ay - 22;
                    $(aD).css({
                        height: (aB + "px")
                    });
                    $(aa).css({
                        height: (aB + "px")
                    });
                    if (aB >= 0 && ax >= 0) {
                        if (ax <= aB + 22) {
                            $(aD).hide()
                        } else {
                            $(aD).show()
                        }
                        if (am != (ax / aB)) {
                            am = ax / aB;
                            T(am);
                            aE(0)
                        }
                        var aK = 0;
                        if (aq) {
                            if (aq.offsetTop + aq.scrollHeight >= ax) {
                                aK = 1
                            } else {
                                if (aq.offsetTop + aq.scrollHeight <= ay) {
                                    aK = 0
                                } else {
                                    aK = aq.offsetTop / ax
                                }
                            }
                            aE(aK);
                            var aN = ao(M * aK, aw);
                            if (aN > aB - P) {
                                aN = aB - P
                            }
                        }
                        if (N) {
                            aE(N);
                            var aN = ao(M * N, aw);
                            if (aN > aB - P) {
                                aN = aB - P
                            }
                        }
                    }
                };
                ah = setInterval(this.render, 50);
                $(aD).empty();
                if (ae && ab.offsetHeight == J.offsetHeight) {
                    var aC = false;
                    var H = false;
                    aD.appendChild(ab);
                    aD.appendChild(J);
                    $(ab).on("mousedown", function() {
                        at();
                        aC = true
                    });
                    $(J).on("mousedown", function() {
                        aH();
                        H = true
                    });
                    $(ab).on("mouseup", function() {
                        $(aD).removeClass("bdbriscroll-ctrl-scroll-touch");
                        aC = false
                    });
                    $(J).on("mouseup", function() {
                        $(aD).removeClass("bdbriscroll-ctrl-scroll-touch");
                        H = false
                    });
                    $(document).on("mouseup", ad)
                }
                aD.appendChild(aa);
                aD.appendChild(L);
                L.appendChild(Z);
                L.appendChild(aj);
                L.appendChild(aJ);
                L.onDragstart = function() {
                    return false
                };
                $(L).on("mouseover", function() {
                    $(L).addClass("bdbriscroll-slider-hover");
                    $(aD).addClass("bdbriscroll-ctrl-scroll-hover")
                });
                $(L).on("mousedown", function() {
                    $(L).addClass("bdbriscroll-slider-touch");
                    $(aD).addClass("bdbriscroll-ctrl-scroll-touch")
                });
                $(L).on("mouseout", function() {
                    $(L).removeClass("bdbriscroll-slider-hover")
                });
                $(L).on("mouseup", function() {
                    $(L).removeClass("bdbriscroll-slider-touch")
                });
                $(aD).on("mouseover", function() {
                    $(aD).addClass("bdbriscroll-ctrl-scroll-hover")
                });
                $(aD).on("mousedown", function() {
                    $(aD).addClass("bdbriscroll-ctrl-scroll-touch")
                });
                $(aD).on("mouseout", function() {
                    $(aD).removeClass("bdbriscroll-ctrl-scroll-hover")
                });
                $(aD).on("mouseup", function() {
                    $(aD).removeClass("bdbriscroll-ctrl-scroll-touch")
                });
                $(aa).on("click", X);
                if (az&&!this.onwheel) {
                    if (!$(O).hasClass("bdbriscroll-onwheel")) {
                        $(O).on("DOMMouseScroll", U);
                        $(O).on("mousewheel", U);
                        $(O).addClass("bdbriscroll-onwheel")
                    }
                }
                if (O) {
                    $(O).on("scroll", function() {
                        if (!I) {
                            aE(O.scrollTop / (O.scrollHeight - O.offsetHeight), 1)
                        }
                    })
                }
                $(L).on("mousedown", function(aK) {
                    Y = document.onselectstart;
                    document.onselectstart = function() {
                        return false
                    };
                    G = window.setInterval(S, 40);
                    $(ar).css({
                        "-moz-user-select": "none"
                    });
                    $(ar).css({
                        "-webkit-user-select": "none"
                    });
                    ap = aK.clientY - L.offsetTop;
                    $(document).on("mousemove", K);
                    $(document).on("mouseup", aF);
                    I = 1;
                    aK.preventDefault();
                    return false
                });
                function ao(aL, aM, aK) {
                    if (aK) {
                        aL = aL > aK ? aK : aL
                    }
                    return aL >= aM ? aL : aM
                }
                function S() {
                    aA.call(window, {
                        value: ag,
                        scale: am
                    })
                }
                function at() {
                    if (aG) {
                        clearInterval(aG)
                    }
                    ai();
                    aG = setInterval(function() {
                        if (aC) {
                            ai()
                        } else {
                            clearInterval(aG)
                        }
                    }, 100)
                }
                function aH() {
                    if (au) {
                        clearInterval(au)
                    }
                    af();
                    au = setInterval(function() {
                        if (H) {
                            af()
                        } else {
                            clearInterval(au)
                        }
                    }, 100)
                }
                function ai() {
                    var aK = ag - an;
                    aK = (aK < 0) ? 0 : aK;
                    aE(aK)
                }
                function af() {
                    var aK = ag + an;
                    aK = (aK > 1) ? 1 : aK;
                    aE(aK)
                }
                function K(aK) {
                    aK = window.event || aK;
                    var aL = ao(aK.clientY - ap, aw, R);
                    ag = (aL - aw) / (R - aw);
                    $(L).css({
                        top: (aL + "px")
                    });
                    return false
                }
                function aF() {
                    $(aD).removeClass("bdbriscroll-ctrl-scroll-hover");
                    $(aD).removeClass("bdbriscroll-ctrl-scroll-touch");
                    $(L).removeClass("bdbriscroll-slider-hover");
                    $(L).removeClass("bdbriscroll-slider-touch");
                    $(ar).css({
                        "-moz-user-select": ""
                    });
                    $(ar).css({
                        "-webkit-user-select": ""
                    });
                    if (G) {
                        window.clearInterval(G)
                    }
                    if (Y) {
                        document.onselectstart = Y
                    } else {
                        document.onselectstart = function() {
                            return true
                        }
                    }
                    $(document).unbind("mousemove", K);
                    $(document).unbind("mouseup", aF);
                    $(L).addClass("bdbriscroll-slider OP_LOG_BTN");
                    I = 0;
                    return false
                }
                function X(aK) {
                    aE((aK.offsetY || aK.layerY) / aB)
                }
                function aE(aM, aK) {
                    aM = aM < 0 ? 0 : aM;
                    aM = aM > 1 ? 1 : aM;
                    ag = aM;
                    var aL = (R - aw) * ag + aw;
                    $(L).css({
                        top: (aL + "px")
                    });
                    if (!aK) {
                        S()
                    }
                }
                function U(aK) {
                    aK.preventDefault();
                    aK = aK.originalEvent;
                    if (aK) {
                        this.onwheel = 1;
                        var aN = ( - aK.wheelDelta || (aK.detail && aK.detail * 40) || 0) / al;
                        var aM = aN;
                        var aL = aM > 0 ? O.scrollTop + 2: O.scrollTop - 2;
                        $(ar).css({
                            zoom: "1"
                        });
                        if (aL > 0 && aL < (ar.offsetHeight - O.offsetHeight)) {
                            O.scrollTop += aM;
                            ag = O.scrollTop / (O.scrollHeight - O.offsetHeight)
                        } else {
                            if (!Q || $(aD).css("display") == "none") {
                                document.documentElement.scrollTop += aM;
                                document.body.scrollTop += aM
                            }
                        }
                    }
                }
                function T(aK) {
                    am = (aK > 10) ? 10 : aK;
                    if (am <= 1) {
                        $(L).css({
                            display: "none"
                        });
                        return 
                    }
                    $(L).css({
                        display: "block"
                    });
                    var aL = aB - 2 * av;
                    P = parseInt(aL / am);
                    P = (P < 15) ? 15 : P;
                    R = aB - av - P;
                    $(L).css({
                        height: (P + "px")
                    })
                }
                if (am > 1) {
                    T(am)
                }
                this.dispose = function() {
                    if (Y) {
                        document.onselectstart = Y
                    } else {
                        document.onselectstart = function() {
                            return true
                        }
                    }
                    $(document).unbind("mousemove", K);
                    $(document).unbind("mouseup", aF);
                    $(document).unbind("mouseup", ad);
                    if (G) {
                        clearInterval(G)
                    }
                    if (aG) {
                        clearInterval(aG)
                    }
                    if (au) {
                        clearInterval(au)
                    }
                    if (ah) {
                        clearInterval(ah)
                    }
                }
            }
            var A = $(window).height() - b.offset().top - 34;
            z.height(A);
            f = new C({
                content: z.get(0),
                length: A - 20,
                mousewheellock: true,
                wheeldelta: 5
            });
            $(window).on("resize", function() {
                var G = $(window).height() - b.offset().top - 34;
                z.height(G);
                f && f.render(G - 20)
            })
        }
        i = 600;
        m = ($(window).height() < i) ? i : $(window).height();
        $(window).on("resize", function() {
            m = ($(window).height() < i) ? i : $(window).height();
            v && v.css({
                "min-height": m
            });
            if (!$.support.leadingWhitespace) {
                v && v.css({
                    height: m
                })
            }
        });
        $.each(v.find("a"), function(H, G) {
            $(G).on("mousedown", function() {
                $(G).attr("name") && x($(G).attr("name"))
            })
        })
    }
    $.each($(".bri-btlinks").find("a"), function(A, z) {
        $(z).on("mousedown", function() {
            $(z).attr("name") && x($(z).attr("name"))
        })
    })
});
