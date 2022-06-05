function escapeHTML(a) {
    return a.replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/&/g, "&amp;").replace(/ /g, "&nbsp;").replace(/"/g, "&#34;").replace(/'/g, "&#39;")
}
function initPreload(a2) {
    document.write = document.writeln = function() {};
    function D() {
        Cookie.set("ISSW", "1", null, null, new Date(new Date().getTime() + 300 * 1000))
    }
    if (bds && bds.comm && bds.comm.query == "clearissw") {
        Cookie.clear("ISSW")
    }(function() {
        var bf = $.Deferred();
        bds.comm.registerUnloadHandler = function(bg) {
            bf.done(bg)
        };
        bds.comm.resolveUnloadHandler = function() {
            bf.resolve();
            bf = $.Deferred()
        }
    })();
    function aX(bf) {
        if (bf && typeof bf == "string") {
            bf = $.parseJSON(bf)
        }
        if (bf && bf.length) {
            $.each(bf, function(bh, bi) {
                if (bi.indexOf(a5.protocol) === 0) {
                    var bg = new Image();
                    bg.src = bi
                }
            })
        }
    }
    function aT(bf) {
        return $.trim(bf).replace(/\s+/g, " ")
    }
    function ap(bh) {
        if (typeof bh == "string") {
            var bf, bg = 0;
            for (bf = 0; bf < bh.length; bf++) {
                bg += bh.charCodeAt(bf)
            }
            return bg
        }
        return 0
    }
    function bc(bk) {
        var bf = {};
        var bj, bg, bm, bh;
        if (bk.indexOf("?")>-1) {
            bm = bk.split("?");
            bh = bm[1]
        } else {
            bh = bk
        }
        if (bh.indexOf("&")>-1) {
            bj = bh.split("&")
        } else {
            bj = new Array(bh)
        }
        for (var bi = 0; bi < bj.length; bi++) {
            try {
                bj[bi] = bj[bi].indexOf("=")>-1 ? bj[bi] : bj[bi] + "=";
                bg = bj[bi].split("=");
                bf[bg[0]] = decodeURIComponent(bg[1].replace(/\+/g, " "))
            } catch (bl) {}
        }
        return bf
    }
    window.b_rec = function(bh) {
        var bf;
        if (bh) {
            bf = navigator.userAgent
        } else {
            try {
                bf = (window.external && window.external.twGetRunPath) ? window.external.twGetRunPath() : ""
            } catch (bg) {
                bf = ""
            }
        }
        bf = bf.replace(/:/, "~").replace(/\t/, "`");
        return bf
    };
    window.scr_rec = function() {
        var bf = "";
        try {
            bf += [document.body.clientWidth, document.body.clientHeight, window.screenTop, window.screenLeft, window.screen.height, window.screen.width].join("_")
        } catch (bg) {}
        return bf
    };
    window.reh_rec = function() {
        var bh = [], bf = [];
        try {
            $("#content_left").children(".result,.result-op").each(function(bi, bj) {
                bh.push($(bj).height())
            })
        } catch (bg) {}
        try {
            $("#con-ar").children(".result,.result-op").each(function(bi, bj) {
                bf.push($(bj).height())
            })
        } catch (bg) {}
        return bh.join("_") + "|" + bf.join("_")
    };
    window.onerror = function() {
        if (window.console && console.debug) {
            console.debug(arguments)
        }
        bds.comm.jserror = Array.prototype.slice.call(arguments).join("\t");
        at(bds.comm.jserror)
    };
    window.hash = function(bg, bf) {
        if (!bg) {
            return 
        }
        if (bg&&!bf && W) {
            return W.p(bg)
        }
        if (bg && bf && W) {
            W.p(bg, bf);
            a5.href = W.buildSearchUrl()
        }
    };
    var ae, be = false;
    function p(bf) {
        if (!be) {
            var bf = $.extend({
                top: 105,
                "z-index": 80
            }, bf);
            if (!ae) {
                ae = $("<div id='_mask'/>")
            }
            ae.css({
                opacity: 0.3,
                position: "absolute",
                background: "#fff",
                "z-index": bf["z-index"],
                top: bf.top + "px",
                left: "0"
            });
            be = true;
            ae.width(t.width());
            ae.height(t.height() - bf.top);
            ae.appendTo(t)
        }
    }
    function aH() {
        if (ae && be) {
            be = false;
            ae.remove()
        }
    }
    function l(bf, bi, bg) {
        bg || (bg = 0);
        var bh = bf.length;
        if (bg < 0) {
            bg = bh + bg
        }
        for (; bg < bh; bg++) {
            if (bf[bg] === bi) {
                return bg
            }
        }
        return - 1
    }(function() {
        var bf = $.globalEval;
        $.globalEval = function(bg) {
            var bi = new Date().getTime();
            try {
                bf.apply($, arguments)
            } catch (bh) {}
            if (new Date().getTime() - bi > 500) {}
        }
    })();
    function H(bj, bi, bk) {
        var bh = bi.find("script:not([src])"), bg = 0;
        var bf = $.globalEval;
        $.globalEval = function(bl) {
            window.currentScriptElem = bh[bg];
            bg++;
            try {
                bf.apply($, arguments)
            } catch (bm) {
                if (window.console && console.debug) {
                    console.debug(bl);
                    console.debug(bm)
                }
            }
        };
        if (bk == "insertBefore") {
            bi.insertBefore(bj)
        } else {
            bj.append(bi)
        }
        window.currentScriptElem = undefined;
        $.globalEval = bf
    }
    function a8(bf) {
        try {
            bf()
        } catch (bg) {
            if (window.console && console.debug) {
                console.debug(bg)
            }
            at(bg.toString())
        }
    }
    var at = (function() {
        var bf;
        return function(bg) {
            if (bds && bds.comm && bds.comm.js_error_monitor) {
                bf = new Image();
                bf.src = bds.comm.js_error_monitor + "?" + $.param({
                    url: a5.href,
                    time: bds.comm.serverTime,
                    explore: navigator.userAgent,
                    info: bg,
                    info_type: 1
                })
            }
        }
    })();
    var ag = {
        use_cache_repeatedly: true,
        index_form: "#form",
        kw: "#kw",
        result_form: "#form"
    };
    if (a2) {
        $.extend(ag, a2)
    }
    var aS = 15;
    var bb = 60000;
    var aN = window.__confirm_timeout ? window.__confirm_timeout: 10000;
    var i = (function() {
        var bf = [];
        function bg(bh) {
            if (typeof bh != "object" || bh == null) {
                return 
            }
            if (bh.xhr && bh.xhr.abort&&!bh.env.equals(W)) {
                bh.xhr.abort()
            }
            if (bh.base64) {
                bh.base64.destroy()
            }
            if (bh.pdc) {
                bh.pdc.destroy()
            }
            delete bh.xhr;
            delete bh.html
        }
        return {
            find: function(bh) {
                return $.grep(bf, bh)
            },
            getCacheList: function() {
                var bh = $.map(bf, function(bi) {
                    if (bi && (new Date().getTime() - bi.startTime > bb)) {
                        return false
                    } else {
                        return bi.querySign
                    }
                });
                bh = $.grep(bh, function(bi) {
                    return !!bi
                });
                return bh.join("\t")
            },
            hasCache: function(bj, bi) {
                if (!bi) {
                    bi = {}
                }
                var bh = bk(bj);
                if (bh && (new Date().getTime() - bh.startTime > bb)) {
                    this.deleteCache(bh);
                    bh = null
                }
                return bh;
                function bk(bl) {
                    var bm, bn;
                    bn = bl.p("wd");
                    if (!bn) {
                        return false
                    }
                    $.grep(bf, function(bo) {
                        if (bi.loaded&&!bo.loaded) {
                            return false
                        }
                        if (bo.real_wd ? (bl.equals(bo.env.clone({
                            wd: bo.real_wd
                        }))) : (bl.equals(bo.env))) {
                            bm = bo
                        }
                    });
                    if (bm) {
                        return bm
                    }
                    return null
                }
            },
            shouldShow: function(bh) {
                if (bh.force) {
                    return true
                }
                if (!bh.shouldShow&&!bh.force && bh.no_predict) {
                    return false
                }
                var bi = aT(kw.val());
                if (!bi || (ao && bh.env.equals(ao.env))) {
                    return false
                }
                if (bh.env.p("wd").indexOf(bi) == 0) {
                    return true
                }
                if (bh.real_wd.indexOf(bi) == 0) {
                    return true
                }
                return false
            },
            getCacheBySign: function(bi) {
                var bh = false;
                $.each(bf, function(bk, bj) {
                    if (!bh && bj.loaded && bj.querySign == bi && (!bj.env.p("pn") || bj.env.p("pn") == 0)) {
                        bh = bj
                    }
                });
                return bh
            },
            addCache: function(bh) {
                if (l(bf, bh)!=-1) {
                    return 
                }
                if (bh.env.p("srcid") || bh.env.p("cq")) {
                    return 
                }
                bf.unshift(bh);
                while (bf.length > aS) {
                    bg(bf.pop())
                }
            },
            deleteCache: function(bh) {
                bg(bh);
                bf = $.grep(bf, function(bi) {
                    return bi !== bh
                })
            },
            deleteCacheByEnv: function(bh) {
                bf = $.grep(bf, function(bj) {
                    var bi = bj.env.equals(bj.env);
                    if (bi) {
                        bg(bj)
                    }
                    return !bi
                })
            },
            clearCache: function() {
                bf = $.grep(bf, function(bi, bh) {
                    if (bh !== ao) {
                        bg(bh);
                        return false
                    } else {
                        return true
                    }
                });
                bf = []
            }
        }
    })();
    var a5 = document.location;
    var ak = {
        onurlchange: function() {}
    };
    (function() {
        var bj = "onhashchange" in window;
        var bg = "onpopstate" in window;
        if (window.__disable_popstate) {
            bg = false
        }
        var bl = a5.pathname.length > 1 ? a5.pathname: "/s";
        if (navigator.userAgent.match(/MSIE (6|7)/)) {
            bj = false;
            bg = false
        }
        if (ag.disable_popstate) {
            bg = false
        }
        if (!bj&&!bg) {
            D()
        }
        function bi() {
            var bo = a5.href.match(/#+(.*)$/);
            return bo ? bo[1] : ""
        }
        var bf = (function() {
            var bp = "", bo;
            return function(br, bq) {
                if (bq) {
                    bp = bq.buildQueryString();
                    a5.hash = bp
                }
                if (br || bp != bi()) {
                    bk(br);
                    bp = bi()
                } else {
                    aH()
                }
            }
        })();
        ak.setUrl = function(bo) {
            if (bg) {
                bh(false, bo)
            } else {
                if (bj) {
                    bf(false, bo)
                }
            }
        };
        function bm() {
            var bo = a5.href.match(/\?([^#]+)/);
            return bo ? bo[1].replace(/\?/g, "&") : ""
        }
        function bn(bo) {
            if (/wd=/.test(bo)) {
                return bo.replace(/&rsv[^=]*=[^&]*/g, "").replace(/[^a-zA-Z0-9]url=/g, "")
            }
        }
        function bk(bo) {
            var bp = new P(bc(ak.getQueryString()));
            if (!bp.hashCode()) {
                if (pageState != 0) {
                    a5.replace(a5.pathname + a5.search.replace(/([?&])isidx=[^&*]&?/, "$1"))
                } else {
                    if (a5.search != a5.search.replace(/([?&])isidx=[^&*]&?/, "$1")) {
                        a5.replace(a5.pathname + a5.search.replace(/([?&])isidx=[^&*]&?/, "$1"))
                    }
                }
            } else {
                if (pageState == 0) {
                    an(bp)
                }
            }
            ak.onurlchange(bp, bo)
        }
        var bh = (function() {
            var bo = "", bp;
            return function(br, bq) {
                if (bq) {
                    bo = bq.buildQueryString();
                    window.history.pushState(bq, "", bq.buildSyncSearchUrl())
                }
                if (br || bo != bm()) {
                    bk(br);
                    bo = bm()
                } else {
                    aH()
                }
            }
        })();
        ak.getQueryString = function() {
            if (bg) {
                return bm()
            } else {
                if (/wd=/.test(bi())) {
                    return bi()
                } else {
                    return bm()
                }
            }
        };
        ak.init = function() {
            if (bg) {
                (function() {
                    var bp = a5.href;
                    var bq = false;
                    $(window).on("swap_begin", function() {
                        bq = true
                    });
                    $(window).bind("popstate", function() {
                        if (bq ||!bp || bp != a5.href) {
                            bh()
                        }
                        bp = null
                    });
                    $(window).bind("hashchange", function() {
                        var br = bi();
                        if (/wd=/.test(br)) {
                            a5.replace(bl + "?" + br)
                        }
                    })
                })()
            } else {
                if (bj) {
                    $(window).bind("hashchange", function() {
                        bf()
                    });
                    $(function() {
                        bf()
                    })
                }
            }
            var bo = bi();
            if (/wd=/.test(bo)) {
                if (bg) {
                    window.history.replaceState(null, "", bl + "?" + bo);
                    bh()
                } else {
                    if (bj) {
                        bf()
                    } else {
                        a5.replace(bl + "?" + bo)
                    }
                }
            }
        };
        ak.support = function() {
            return (bg || bj) && Cookie.get("ISSW") != 1&&!window.__disable_preload
        };
        ak.getLinkParams = function() {
            if (!bg) {
                var bo = bi();
                return bn(bo)
            }
            if (a5.protocol === "https:") {
                var bp = bm();
                if (!bp) {
                    bp = bi()
                }
                return bn(bp)
            }
            return false
        };
        ak.clickResultLink = function(bo, bq, bp) {
            if (bg) {
                window.history.pushState(null, "", new P(bp).buildSyncSearchUrl());
                bh();
                return false
            } else {
                bo.attr("href", bq.buildSearchUrl(bp)).attr("target", "_self")
            }
        };
        ak.submit = function(bp, bo) {
            setTimeout(function() {
                if (bg) {
                    window.history.pushState(bp, "", bp.buildSyncSearchUrl());
                    bh(bo)
                } else {
                    if (bj) {
                        a5.href = bp.buildSearchUrl();
                        bf(bo)
                    } else {
                        a5.href = bp.buildSyncSearchUrl()
                    }
                }
            }, 0)
        };
        window.changeUrl = function(bp) {
            var bo = new P(bc(bp));
            ak.submit(bo, true)
        }
    })();
    ak.onurlchange = function(bg, bf) {
        R.done(function() {
            aP.setKey(bg.p("wd"));
            aP.hide()
        });
        aL = new Date().getTime();
        kw.val(bg.p("wd"));
        aQ("");
        q({
            env: bg,
            force: true,
            use_cache: !bf,
            no_predict: true
        })
    };
    var am = ag.disable ? ag.disable: false;
    if (window.__disable_preload) {
        am = true
    }
    var N = am;
    var y = false;
    if (window.__disable_predict) {
        y = true
    }
    var a4 = y;
    var g = 200;
    var aV = 250;
    var aB = 2000;
    var aI = 100;
    var av = 800;
    var aG = bds.comm.switchAddMask ? bds.comm.switchAddMask: false;
    if (!aG) {
        aG = window.__switch_add_mask ? window.__switch_add_mask : false
    }
    aG = true;
    var f = bds.comm.preloadMouseMoveDistance ? bds.comm.preloadMouseMoveDistance: 5;
    var x = 0;
    var ay = function() {};
    var aj = bc(a5.search);
    if (!ak.support()) {
        !function() {
            function bf() {
                a5.hash && a5.hash.match(/[^a-zA-Z0-9](wd|word)=/) && a5.replace("//www.baidu.com/s?" + a5.href.match(/#(.*)$/)[1])
            }
            a5.hash.match(/[^a-zA-Z0-9](wd|word)=/) ? (a5.replace("//www.baidu.com/s?" + a5.href.match(/#(.*)$/)[1]), (function() {
                throw new Error("redirect to sync")
            })()) : (document.getElementById("wrapper").style.display = "block", "onhashchange" in window ? window.onhashchange = bf : setInterval(bf, 200))
        }();
        N = am = true
    }
    var F = Cookie.get("BAIDUID", "nobdid").split(":")[0];
    var au = F.substr(0, 6) + F.substr(F.length - 5, 5) + parseInt(Math.random() * 99999);
    while (au.length < 16) {
        au += "0"
    }
    au = encodeURIComponent(au);
    var al, a9;
    var a6 = index_kw = kw = $(ag.kw), V, a0;
    var aL;
    var T, ad, O, aO, U, az;
    var aa = $("#wrapper_wrapper");
    var aZ = [];
    var K = window.__async_strategy;
    al = $(ag.index_form);
    if (al.attr("target") == "_blank") {
        window.__disable_index_predict = true;
        N = am = true
    }
    var a3 = al.serializeArray();
    a9 = $(ag.result_form);
    var aR = new Date().getTime();
    function P(bj) {
        if (!P.__init) {
            P.__init = true;
            var bl = ["wd", "pn", "nojc", "cl", "cq", "srcid"];
            var bh = ["wd", "cl", "ct", "tn", "rn", "ie", "f", "lm", "si", "usm", "z", "ch", "sts", "vit", "dsp", "trh", "trb", "tre", "la", "lo", "st", "nojc", "haobd", "rtt", "bsst", "gvideo", "__eis", "__eist", "oq", "fenlei", "sid", "rsv_idx"];
            var bi = ["w", "word"];
            P.prototype.clone = function(bn) {
                var bo = new P(bk(this.params));
                if (typeof bn == "object") {
                    for (var bm in bn) {
                        if (bn.hasOwnProperty(bm) && l(bh, bm) >= 0) {
                            bo.p(bm, bn[bm])
                        }
                    }
                }
                return bo
            };
            P.prototype.buildSearchUrl = function(bm) {
                return a5.protocol + "//" + a5.host + a5.pathname + a5.search + "#" + this.buildQueryString(bm)
            };
            P.prototype.buildSyncSearchUrl = function(bm) {
                return a5.protocol + "//" + a5.host + "/s?" + this.buildQueryString(bm)
            };
            P.prototype.buildQueryString = function(bo) {
                var bn = bk(this.params);
                if (typeof bo == "object") {
                    for (var bm in bo) {
                        if (bo.hasOwnProperty(bm)) {
                            bn[bm] = bo[bm]
                        }
                    }
                }
                var bp = "";
                for (param in bn) {
                    if (param && bn.hasOwnProperty(param) && bn[param] !== "") {
                        bp += param + "=" + encodeURIComponent(bn[param]).replace(/'/g, "%27") + "&"
                    }
                }
                return bp.substr(0, bp.length - 1)
            };
            P.prototype.equals = function(bn) {
                if (!bn ||!bn.p) {
                    return false
                }
                for (var bo = 0; bo < bl.length; bo++) {
                    var bm = bl[bo];
                    if (bm == "pn") {
                        var bq = this.p(bm) ? this.p(bm): "0";
                        var bp = bn.p(bm) ? bn.p(bm): "0";
                        if (bq != bp) {
                            return false
                        }
                    } else {
                        if (this.p(bm) != bn.p(bm)) {
                            return false
                        }
                    }
                }
                return true
            };
            P.prototype.p = function(bn, bm) {
                if (l(bi, bn) >= 0) {
                    bn = "wd"
                }
                if (bm === undefined) {
                    return this.params[bn]
                }
                this.params[bn] = bm;
                return this
            };
            P.prototype.hashCode = function() {
                var bn = [];
                if (!this.p("wd")) {
                    return ""
                }
                for (var bo = 0; bo < bl.length; bo++) {
                    var bm = bl[bo];
                    if (bm == "pn"&&!this.p(bm)) {
                        bn.push("0")
                    } else {
                        bn.push(this.p(bm))
                    }
                }
                return bn.join("\t")
            };
            P.prototype.filterOtherParams = function() {
                for (var bm in this.params) {
                    if (this.params.hasOwnProperty(bm) && l(bh, bm) < 0) {
                        delete this.params[bm]
                    }
                }
            };
            P.prototype.wdSameName = function() {
                var bm;
                for (bm = 0; bm < bi.length; bm++) {
                    if (this.params && this.params[bi[bm]]) {
                        this.params.wd = this.params[bi[bm]];
                        delete this.params[bi[bm]]
                    }
                }
            }
        }
        this.params = {};
        if (typeof bj == "object") {
            for (var bf in bj) {
                if (bj.hasOwnProperty(bf)) {
                    this.p(bf, bj[bf])
                }
            }
        }
        for (var bg = 0; bg < a3.length; bg++) {
            if (!this.p(a3[bg].name)) {
                this.p(a3[bg].name, a3[bg].value)
            }
        }
        this.wdSameName();
        function bk(bn) {
            if (typeof bn == "object") {
                var bm = {};
                for (bf in bn) {
                    if (bn.hasOwnProperty(bf)) {
                        bm[bf] = bn[bf]
                    }
                }
            } else {
                bm = bn
            }
            return bm
        }
    }
    window.pageState = 0;
    var W = null;
    var ao = null;
    var aY = document.location.href;
    var ah = false;
    var aP, aC, j;
    var R = $.ajax({
        dataType: "script",
        cache: true,
        url: window._is_bdssug_sam ? "http://s1.bdstatic.com/r/www/cache/static/sug/js/bdsug_input_event_sam_344c1a25.js": "http://s1.bdstatic.com/r/www/cache/static/sug/js/bdsug_input_event_58eaba58.js"
    });
    var af;
    var n = "focus";
    var J;
    (function() {
        window.PDC_ASYNC = {
            setParam: function(bg, bh) {
                if (ao && ao.pdc) {
                    ao.pdc.setParam(bg, bh)
                }
            }
        };
        var bf = window.PDC_ASYNC.log = {}
    })();
    function aw(bA) {
        var bu = {
            product_id: 45,
            page_id: 317,
            page_type: 0
        }, bj = {}, bE = {
            st: 0,
            pt: 0,
            net: 0,
            dom: 0,
            fs: 0
        }, bx = [], bg = $.Callbacks(), bh = {}, bn = null, bk = null, bl = 600;
        function bq(bF) {
            if (typeof bF === "string") {
                bE[bF] = Date.now ? Date.now() : + new Date()
            }
        }
        function by(bF, bG) {
            if (typeof bF === "string") {
                bj[bF] = bG
            }
        }
        function bs() {
            bj.cus_net = bE.net > bE.st && bE.net - bE.st - bj.cus_srv > 0 ? bE.net - bE.st - bj.cus_srv : 1;
            bj.cus_tti2 = bE.dom > bE.st ? bE.dom - bE.st : 1;
            bj.cus_frdom = bE.dom - bE.pt;
            bj.cus_fs = bE.fs > bE.st ? bE.fs - bE.st : bj.cus_tti2;
            bj.cus_frext = bj.cus_fs - bj.cus_tti2
        }
        function br(bG) {
            var bH = "";
            for (var bF in bG) {
                if (bF && bG.hasOwnProperty(bF) && bG[bF] !== "") {
                    bH += "&" + bF + "=" + encodeURIComponent(bG[bF])
                }
            }
            return bH
        }
        function bw(bG) {
            var bG = [];
            for (var bH in bh) {
                bG.push(bh[bH])
            }
            var bF = bn = $.when.apply($, bG);
            bn.always(function() {
                if (bF !== bn) {
                    return 
                }
                bB()
            })
        }
        function bv() {
            var bG = Array.apply(null, arguments);
            if (!bG.length > 0) {
                return 
            }
            for (var bF = 0; bF < bG.length; bF++) {
                bh[bG[bF]] = $.Deferred()
            }
        }
        function bo() {
            by("qid", bA.qid);
            by("cus_q", (bA.real_wd || bA.env.p("wd")));
            by("sid", bds.comm.sid);
            by("cus_newindex", bds.comm.newindex);
            by("supportis", bds.comm.supportis)
        }
        function bD() {
            bA = null;
            bk = null
        }
        function bt(bF) {
            bh[bF].resolve();
            if (bF == "swap_end") {
                setTimeout(function() {
                    bt("swap_end_5s")
                }, 5000)
            }
        }
        function bm() {
            bv("swap_end", "swap_end_5s");
            if (bds.comm.supportis) {
                bv("confirm")
            }
            bw()
        }
        function bz() {
            bm();
            bE.st = 0;
            bE.fs = 0;
            bE.dom = 0
        }
        function bB() {
            var bM = Math.random(), bH = /9179|9180|8500|9999|9989|8967|8968|9449|9450|9900|9901/, bI = /7744|7745/, bN = bM > 0.51 && bM < 0.52;
            if (((bM > 0.51 && bM < 0.52) || (bH.test(bds.comm.sid) && (bM > 0 && bM < 0.5)) || (bI.test(bds.comm.sid) && (bM > 0 && bM < 0.2)) || bds.comm.intrSid)) {
                if ((bH.test(bds.comm.sid) || bI.test(bds.comm.sid))) {
                    if (!bN) {
                        by("issam", 1)
                    } else {
                        by("issam", 2)
                    }
                }
                bs();
                bp(bk);
                bi(bk);
                bo();
                bg.fire();
                var bL = "//www.baidu.com/nocache/fesplg/s.gif?log_type=sp", bF = "";
                bF += br(bu) + br(bj);
                var bG = bL + bF, bK = new Image(), bJ = "_LOG_" + new Date().getTime();
                bK.onload = function() {
                    delete window[bJ]
                };
                window[bJ] = bK;
                bK.src = bG
            }
        }
        function bf(bF) {
            bk = bF;
            bF.find("img").one("load", function() {
                var bK = $(this).offset(), bI = bK.top, bJ = bK.left, bH = "";
                if (bI < bl && bI > 0) {
                    bq("fs");
                    var bG = bE.fs - bE.dom;
                    bx.push(bI + "_" + bJ + "_" + bG);
                    if ($(this).attr("data-src") || /^http/.test($(this).attr("src"))) {
                        bH = $(this).attr("data-src") || $(this).attr("src")
                    } else {
                        bH = "base64"
                    }
                    by("ic_lis", bH)
                }
            })
        }
        function bp(bG) {
            var bL = 0, bK = bG.find("img"), bJ = bG.find("#content_left").find("img"), bM = 0, bF = 0, bI = 0;
            for (var bH = 0; bH < bK.length; bH++) {
                bI = bK.eq(bH).offset().top;
                if (bI < bl && bI > 0) {
                    bL++
                }
            }
            by("cus_ic", bK.length);
            by("cus_extic", bL);
            by("cus_extlic", bM);
            by("cus_icl", bJ.length);
            by("cus_icr", bG.find("#content_right").find("img").length);
            by("img_info", bx.join(","));
            by("psize", bG.html().length)
        }
        function bi(bF) {
            var bK = {}, bJ = [], bI = bF.find("#content_left,#con-ar").children("*[tpl]"), bG = "";
            if (bI.length > 0) {
                for (var bH = 0; bH < bI.length; bH++) {
                    bG = bI.eq(bH).attr("tpl");
                    if (!bK.hasOwnProperty(bG)) {
                        bK[bG] = 1;
                        bJ.push(bG)
                    }
                }
            }
            if (bJ.length > 0) {
                by("tplp", bJ.join("|"))
            }
        }
        function bC(bF) {
            bg.add(bF)
        }
        bm();
        return {
            trigger: bt,
            mark: bq,
            setParam: by,
            onSendlog: bC,
            bindImgLoad: bf,
            destroy: bD,
            init: bz
        }
    }
    function aM(bj) {
        function bh() {
            if (bk != 1, (bk = 1, bm()), bk == 1) {
                var bp = new Date(), bq = false, br = function() {
                    var bu = new Date(), bs = bu - bp - bj, bt = bi();
                    0 > bs && (bs = 0);
                    if (!bt&&!bq) {
                        bf[bn] = bs;
                        bn = (bn + 1)%20
                    }
                    bq = bt;
                    1 == bk && (bp = bu, bg = setTimeout(br, bj))
                };
                bg = setTimeout(br, bj)
            }
        }
        function bm() {
            window.clearTimeout(bg)
        }
        function bi() {
            var bq = ["webkit", "moz", "ms", "o"];
            if ("hidden" in document) {
                return document.hidden
            }
            for (var bp = 0; bp < bq.length; bp++) {
                if (bq[bp] + "Hidden" in document) {
                    return document[bq[bp] + "Hidden"]
                }
            }
            return false
        }
        function bl() {
            bf.slice(bn).concat(bf.slice(0, bn))
        }
        function bo(bx) {
            var bv = 0, bw = Math.max.apply(null, bx);
            var by = 10, bs = 60, br = 4;
            var bu = Cookie.get("ispeed_lsm"), bq = 0, bp = new Date();
            bp.setTime(bp.getTime() + 30 * 365 * 86400000);
            for (var bt = 0; bt < bx.length; bt++) {
                var bz = bx[bt];
                bv += bz
            }
            bv = Math.round(bv / bx.length);
            if (bw > 1000 || bv > 150) {
                bq = bu ? parseInt(bu) : by;
                if (bq < bs) {
                    bq = bq + by > bs ? bs : bq + by;
                    Cookie.set("ispeed_lsm", bq, document.domain, "/", bp)
                }
            } else {
                if (bu && parseInt(bu) >= br) {
                    bq = parseInt(bu) - br;
                    Cookie.set("ispeed_lsm", bq, document.domain, "/", bp)
                }
            }
        }
        var bf = [], bn = 0, bk = 0, bg = false, bj = bj || 250;
        return {
            start: bh,
            stop: function() {
                window.clearTimeout(bg);
                bk = 0
            },
            getData: function() {
                return bf.slice(bn).concat(bf.slice(0, bn))
            },
            monitor: bo
        }
    }
    var aA = null;
    if (bds.comm.supportis) {
        aA = aM();
        aA.start()
    }
    R.done(function() {
        aP = aC = j = bds.se.sug({
            maxNum: 10,
            withoutRich: true,
            withoutZhixin: true,
            form: a9[0],
            ipt: kw[0],
            cbname: "bdsugresultcb",
            submission: ax
        });
        kw.keydown(function(bh) {
            var bi = k(this);
            if (bh.keyCode == 39 && bi == this.value.length && ao && ao.real_wd != this.value && V.text()) {
                kw.val(ao.real_wd);
                aP.check();
                aQ(ao.real_wd)
            }
            if (window.__sam_tip && bh.keyCode == 9 && ao && ao.real_wd != this.value && V.text()) {
                bh.preventDefault();
                kw.val(ao.real_wd);
                aP.check();
                aQ("")
            }
        });
        aP.on("start", function() {
            n = "focus"
        });
        $(window).on("blur", function() {
            aP.hide()
        });
        $(document).on("click", function(bh) {
            if (bh.isTrigger == 2 || bh.isTrigger == 3) {
                return false
            }
            aP.hide()
        });
        var bf, bg;
        aP.on("inputChange", function(bk, bj) {
            if (!bf) {
                bf = kw.val()
            }
            aQ("");
            aW();
            if (pageState == 0 && window.__disable_index_predict) {
                aP.setMaxNum(10);
                return 
            }
            aP.setMaxNum(4);
            var bl = new P({
                pn: "",
                wd: bj.value
            });
            if (pageState == 0 && $.trim(kw.val())) {
                an(bl);
                var bi = $("<div id='ent_sug'>请按“回车”键发起检索</div>").insertBefore("#head");
                $(window).one("swap_begin", function() {
                    bi.remove()
                })
            }
            if (J) {
                J = false;
                return 
            }
            if (window.__restart_confirm_timeout) {
                ac()
            }
            n = "input";
            aL = new Date().getTime();
            if (bg) {
                clearTimeout(bg);
                bg = false
            }
            if ($.trim(bj.value) == "") {
                b();
                return 
            }
            af = bj.checkStore();
            if (!/^[a-zA-Z0-9~!@#$%^&*()_+=-]$/.test(bj.value)) {
                var bh = kw.val();
                if (bf.length > bh.length && bf.indexOf(bh) === 0) {
                    bg = setTimeout(function() {
                        q({
                            env: bl,
                            use_cache: true,
                            force: false
                        });
                        bg = false
                    }, 250)
                } else {
                    q({
                        env: bl,
                        use_cache: true,
                        force: false
                    })
                }
            }
            bf = bh
        });
        aP.on("selectSug", function(bk, bj, bi, bh) {
            aQ("");
            if (pageState == 0 && window.__disable_index_predict) {
                return 
            }
            if (bi==-1) {
                if (ao) {
                    aP.setVisibleSug(ao.real_wd)
                }
                var bl = new P({
                    pn: "",
                    wd: bj.value
                });
                q({
                    env: bl,
                    use_cache: true,
                    force: false
                })
            } else {
                aP.setVisibleSug();
                aQ("");
                var bl = new P({
                    pn: "",
                    wd: bh
                });
                q({
                    env: bl,
                    force: false,
                    use_cache: true,
                    no_predict: true,
                    shouldShow: true
                })
            }
        });
        aP.on("render", function(bi, bh) {
            if (ao) {
                aP.setVisibleSug(ao.real_wd)
            }
        });
        if (pageState == 0) {
            aP.start()
        }
    });
    function k(bh) {
        var bg = 0;
        if (document.selection) {
            var bf = document.selection.createRange();
            bf.moveStart("character", - bh.value.length);
            bg = bf.text.length
        } else {
            if (bh.selectionStart || bh.selectionStart == "0") {
                bg = bh.selectionStart
            }
        }
        return (bg)
    }
    function aE(bf, bg) {
        if (bf) {
            bg = $.extend(bf.log, bg)
        }
    }
    function aJ() {
        if (bds.comm.seinfo) {
            bds.comm.seinfo.rsv_pre = encodeURIComponent(w());
            bds.comm.seinfo.rsv_reh = reh_rec();
            bds.comm.seinfo.rsv_scr = scr_rec();
            c(bds.comm.seinfo)
        }
        if (bds.comm.cgif) {
            var bf = bds.comm.cgifimg = new Image();
            bf.src = bds.comm.cgif
        }(function() {
            var bl = Math.random(), bj = [], bk = function(bn) {
                var bm = $(bn), bq = "", bp;
                for (var bo = 0; bo < bm.length; bo++) {
                    bp = bm.eq(bo);
                    bq = bp.attr("src") || bp.attr("href");
                    if (bi(bq)) {
                        bj.push(encodeURIComponent(bq))
                    }
                }
            }, bi = function(bm) {
                if (/http:\/\//.test(bm)) {
                    return true
                }
                return false
            }, bh = function() {
                var bo = "//www.baidu.com/nocache/fesplg/s.gif?log_type=hm", bp = "";
                bp += "&q=" + bds.comm.query;
                bp += "&error=" + bj.join(",");
                var bm = new Image(), bn = "_HM_LOG_" + new Date().getTime();
                bm.onload = function() {
                    delete window[bn]
                };
                window[bn] = bm;
                bm.src = bo + bp
            }, bg = function() {
                var bo = Math.floor(Math.random() * 4);
                var bm = {
                    www: "https://www.baidu.com/nocache/pdns/az.gif?t=" + ( + new Date()),
                    http: "http://www.baidu.com/nocache/pdns/az.gif?t=" + ( + new Date()),
                    cdn: "https://ss" + bo + ".baidu.com/6ONWsjip0QIZ8tyhnq/ps_default.gif?t=" + ( + new Date()),
                    idc: "https://sp" + bo + ".baidu.com/htpoty.gif?t=" + ( + new Date()),
                    ip: "https://wwwbg.baidu.com/htpoty.gif?t=" + ( + new Date())
                };
                var bn = [], bq = [], bs = {};
                for (var bp in bm) {
                    (function(bw) {
                        var bv = "_SSL_LOG_" + bw + "_" + ( + new Date()), bt = new Image(), bu = new Date();
                        bs[bw] = $.Deferred();
                        bq.push(bs[bw]);
                        bt.onload = function() {
                            bs[bw].resolve();
                            delete window[bv]
                        };
                        bt.onerror = function() {
                            bs[bw].resolve();
                            bn.push(bw + "_" + bo + "=" + (new Date() - bu));
                            delete window[bv]
                        };
                        bt.src = bm[bw]
                    })(bp)
                }
                var br = $.when.apply($, bq);
                br.always(function() {
                    var bv = "//www.baidu.com/nocache/fesplg/s.gif?log_type=hm&type=ssl&", bw = bn.join("&");
                    var bt = new Image(), bu = "_HM_LOG_" + new Date().getTime();
                    bt.onload = function() {
                        delete window[bu]
                    };
                    window[bu] = bt;
                    bt.src = bv + bw
                })
            };
            if (a5.protocol === "https:" && bl < 0.1) {
                bk("img");
                bk("script");
                bk("iframe");
                bk("link");
                if (bj.length > 0) {
                    bh()
                }
            }
            if (bl < 0.01) {
                bg()
            }
        })()
    }
    var L = {};
    window.addPerformanceLog = function(bf, bg) {
        L["perf_" + bf] = bg
    };
    function w() {
        return aZ.length
    }
    var Y, z, a7, s;
    (function() {
        var bi;
        var bm =- 1, bl = 0;
        var bg =- 1, bj =- 1, bf =- 1, bh =- 1;
        var bk = 0;
        Y = function(bo) {
            if (!bo) {
                return 
            }
            bf = bo.pageX;
            bh = bo.pageY;
            bi = bo.target;
            var bn = $(bo.target);
            if (bn.attr("type") == "submit") {
                bk = 1
            }
            var bp = bn.offset();
            bg = bf - bp.left;
            bj = bh - bp.top;
            bl = new Date().getTime()
        };
        z = function(bn) {
            if (!bn || bn.target != bi) {
                return 
            }
            bm = new Date().getTime() - bl
        };
        s = function(bo) {
            if (bds && bds.comm && bds.comm.query) {
                bo = bds.comm.query
            }
            var bn = bk + "." + bm + "." + bg + "." + bj + "." + bf + "." + bh;
            bn = ap(bn + bo) + "." + bn;
            return bn
        };
        a7 = function() {
            bm =- 1;
            bl = 0;
            bg =- 1;
            bj =- 1;
            bf =- 1;
            bh =- 1;
            bk = 0
        }
    })();
    $(document).delegate("a", "mousedown", function() {
        r(ao, ah, 22)()
    });
    function C(bf) {
        $(document).delegate("a", "mousedown", (function() {
            return function() {
                var bl = bf.prefix;
                var bk = $(this);
                var bh = bk.attr("href");
                var bg;
                if (bl && bh && bh.indexOf(bl) == 0) {
                    bh = bh.substring(bl.length)
                }
                if (!bl && bh) {
                    var bi = bh.match(/^http:\/\/[^\/]+/);
                    if (bi) {
                        bl = bi[0]
                    } else {
                        return 
                    }
                    bh = bh.replace(/^http:\/\/[^\/]+/, "")
                }
                if (bh) {
                    bg = bh.match(bf.regex)
                }
                if (bg && bg[2] && bg[2].match(/&(wd|word)=/)) {
                    return 
                }
                if (bh && bg) {
                    if (bf.convertTable && bf.convertTable[bg[1]]) {
                        bg[1] = bf.convertTable[bg[1]]
                    }
                    var bj = ak.getLinkParams();
                    if (bj) {
                        bk.attr("href", bl + "/" + bg[1] + "?" + bg[2] + "&" + bj)
                    }
                }
            }
        })())
    }
    C({
        prefix: "http://www.baidu.com",
        regex: /^\/*(link)\?(.*)$/
    });
    C({
        prefix: "//www.baidu.com",
        regex: /^\/*(link)\?(.*)$/
    });
    C({
        prefix: "http://www.baidu.com",
        convertTable: {
            "baidu.php": "baidu.php",
            "aladdin.php": "aladdin.php",
            "siva.php": "siva.php",
            "adrc.php": "adrc.php",
            "zhixin.php": "zhixin.php"
        },
        regex: /^\/*(baidu\.php|aladdin\.php|siva\.php|adrc\.php|zhixin\.php)\?(.*)$/
    });
    if (a5.host != "www.baidu.com") {
        C({
            prefix: "",
            convertTable: {
                "baidu.php": "baidu.php",
                "aladdin.php": "aladdin.php",
                "siva.php": "siva.php",
                "adrc.php": "adrc.php",
                "zhixin.php": "zhixin.php"
            },
            regex: /^\/*(baidu\.php|aladdin\.php|siva\.php|adrc\.php|zhixin\.php)\?(.*)$/
        })
    }
    C({
        prefix: "http://bzclk.baidu.com",
        regex: /^\/*(adrc\.php)\?(.*)$/
    });
    if (ak.support()) {
        $(document).delegate("a", "click", (function() {
            var bf = a5.protocol + "//" + a5.host;
            return function(bk) {
                var bj = $(this);
                if (bj.attr("target") && bj.attr("target") != "_self") {
                    return 
                }
                var bh = $.trim(bj.attr("href"));
                if (bh && bh.indexOf(bf) == 0) {
                    bh = bh.substring(bf.length)
                }
                if (bh) {
                    matched = bh.match(/^\/*s\?(.*)/)
                }
                if (bh && matched) {
                    var bl = bc(matched[0]);
                    if (!bl.pn) {
                        bl.pn = ""
                    }
                    if (l(["baidurt", "baiduwb", "baidufir", "SE_baiduxueshu_c1gjeupa"], bl.tn) < 0) {
                        var bg = bj.parents("#con-at");
                        if (bg.size() > 0) {
                            p({
                                top: bg.offset().top + bg.height()
                            })
                        }
                        var bi = ak.clickResultLink(bj, W, bl);
                        Y(bk)
                    }
                    return bi
                }
            }
        })())
    }
    $(document).delegate("a", "mouseup", function(bf) {
        z(bf)
    });
    $(document).delegate("#su,#su1", "mouseup", function(bf) {
        z(bf)
    });
    $(document).delegate("#su,#su1", "mousedown", function(bf) {
        Y(bf)
    });
    var t = $("body");
    var Q = document.title;
    (function(bg) {
        var bf;
        bg.fn.textWidth = function() {
            if (!bf) {
                bf = bg('<div data-for="result" style="clear:both;display:block;visibility:hidden;position:absolute;top:0;"><span style="width;inherit;margin:0;font:16px/22px arial;"></span></div>').appendTo("body").find("span")
            }
            bf.html(escapeHTML(bg(this).val()));
            var bh = bf.width();
            return bh
        }
    })(jQuery);
    function a(bf) {
        if (u) {
            if (V) {
                V.html("")
            }
            return 
        }
        if (pageState == 0) {
            return 
        }
        if (window.__disable_kw_tip) {
            return 
        }
        if (!V) {
            a0 = $('<div style="position:absolute;z-index:1;opacity:0.55;width:initial;"><div id="kw_tip" style="display:inline;position:static;color:#666;width:initial;" unselectable="on" onselectstart="return false;" class="s_ipt_tip"/><img style="position:absolute;width:14px;margin:8px 0 0 3px;" src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABoAAAAaCAYAAACpSkzOAAAAAXNSR0IArs4c6QAAAAlwSFlzAAALEwAACxMBAJqcGAAABCJpVFh0WE1MOmNvbS5hZG9iZS54bXAAAAAAADx4OnhtcG1ldGEgeG1sbnM6eD0iYWRvYmU6bnM6bWV0YS8iIHg6eG1wdGs9IlhNUCBDb3JlIDUuNC4wIj4KICAgPHJkZjpSREYgeG1sbnM6cmRmPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5LzAyLzIyLXJkZi1zeW50YXgtbnMjIj4KICAgICAgPHJkZjpEZXNjcmlwdGlvbiByZGY6YWJvdXQ9IiIKICAgICAgICAgICAgeG1sbnM6dGlmZj0iaHR0cDovL25zLmFkb2JlLmNvbS90aWZmLzEuMC8iCiAgICAgICAgICAgIHhtbG5zOmV4aWY9Imh0dHA6Ly9ucy5hZG9iZS5jb20vZXhpZi8xLjAvIgogICAgICAgICAgICB4bWxuczpkYz0iaHR0cDovL3B1cmwub3JnL2RjL2VsZW1lbnRzLzEuMS8iCiAgICAgICAgICAgIHhtbG5zOnhtcD0iaHR0cDovL25zLmFkb2JlLmNvbS94YXAvMS4wLyI+CiAgICAgICAgIDx0aWZmOlJlc29sdXRpb25Vbml0PjI8L3RpZmY6UmVzb2x1dGlvblVuaXQ+CiAgICAgICAgIDx0aWZmOkNvbXByZXNzaW9uPjU8L3RpZmY6Q29tcHJlc3Npb24+CiAgICAgICAgIDx0aWZmOlhSZXNvbHV0aW9uPjcyPC90aWZmOlhSZXNvbHV0aW9uPgogICAgICAgICA8dGlmZjpPcmllbnRhdGlvbj4xPC90aWZmOk9yaWVudGF0aW9uPgogICAgICAgICA8dGlmZjpZUmVzb2x1dGlvbj43MjwvdGlmZjpZUmVzb2x1dGlvbj4KICAgICAgICAgPGV4aWY6UGl4ZWxYRGltZW5zaW9uPjI2PC9leGlmOlBpeGVsWERpbWVuc2lvbj4KICAgICAgICAgPGV4aWY6Q29sb3JTcGFjZT4xPC9leGlmOkNvbG9yU3BhY2U+CiAgICAgICAgIDxleGlmOlBpeGVsWURpbWVuc2lvbj4yNjwvZXhpZjpQaXhlbFlEaW1lbnNpb24+CiAgICAgICAgIDxkYzpzdWJqZWN0PgogICAgICAgICAgICA8cmRmOkJhZy8+CiAgICAgICAgIDwvZGM6c3ViamVjdD4KICAgICAgICAgPHhtcDpNb2RpZnlEYXRlPjIwMTQtMTEtMThUMDA6MTE6NjI8L3htcDpNb2RpZnlEYXRlPgogICAgICAgICA8eG1wOkNyZWF0b3JUb29sPlBpeGVsbWF0b3IgMy4zPC94bXA6Q3JlYXRvclRvb2w+CiAgICAgIDwvcmRmOkRlc2NyaXB0aW9uPgogICA8L3JkZjpSREY+CjwveDp4bXBtZXRhPgpAxNZMAAADDklEQVRIDd1WS2gTURRtvkSQWGNcFlpQo9WWIKi4SxFxIbgyiAs1MX8QYhBdZ6lYcRVpfiSIX+Ku7tyEbktRrFKNn2xiUvxCIVHy9dzpvOFl5jXNom58MLl37jn3nnl35r2XkZH/begGTSgWi22r1+unwHHrdLpp2PFer/cWdr7T6TzNZrMfB+XzmFAoHo/rq9VqFMVnUVjPJ6j8xWazeSafz6+q4ppbjRBEzBBZAPMYx34HvwDhtW63e1iv159jD4BYF9hMMpmknA2HWkgXCoVeoMgJOaOAolfm5ua+8hXcbrfBZrOdBe8+4mbC0MoptPINz+P9PqFgMHgV4F2ZcCOVSt3myWrf6/XuNpvNnyG4HdhaqVTaVSwW22oe3Sv9pxePNtyhIOyzzUSIl8vlvmEmR8nHsDocjkvrrvZXEWo0GjOs77ARLVUcyWQyK/RghCJvFqavSyxLEQLpIgWR9BKz+c4IQ9pbMm/U5/PtFOUoQgClFkDwuYg4KIYP5hPD4duZz1teaIIACP3gCcP4vzE43g7OV1xe6ANF0bq9CjqkY7FYrIyK/J/M5y0vVJCBC7DCF8on8j66oCxu7BRfeIz5vNATOWiNRCJKIiNuZGm7glBCxhewHf0RcRUhfGm0qqWnwdqY93g8FlGCOlar1a4jNkZxCEbVOLtXhIiH/p6WATtW/BLERhlRYHWBQCCK4jdlbDGdTr8S8KSQ5l1gG6LFeo9QCHdR6Fqr1XpYLpd/uVyuLjZcC2JOYNQuJ/FoyFwnOrO8Hun/1QgRjI31PIo96qeK78BbgsgUUDOJYUxjZnRm9Q2+dQqALf+xyWSyI/GBEtQ6qxA5iaJHANGSaOIe61X/Gg86qaYLZ8ST6HyqVCr7UWAMwrS91GCX1UdHOBwex2zeA5dmBtGDaCOdY9LYVIgRh7F+v38CD0TFJTGj0XggkUiUKFfYumGKijjYycuIU9ukNrbb7RV8mfuIu6UzooI0sIPvMRgM9EHQ6dtGqye3dEYkQoP+HaH4Ibh02hpxHcf17wa1DddlUvgLzromHQCKZAIAAAAASUVORK5CYII="/></div>').insertBefore(kw);
            V = a0.find("#kw_tip");
            a0.parent().click(function(bk) {
                var bj = kw.get(0);
                if (bk.target === bj) {
                    return true
                }
                bj.focus();
                var bh = bj.value.length;
                if (document.selection) {
                    var bi = bj.createTextRange();
                    bi.moveStart("character", bh);
                    bi.collapse();
                    bi.select()
                } else {
                    if (typeof bj.selectionStart == "number" && typeof bj.selectionEnd == "number") {
                        bj.selectionStart = bj.selectionEnd = bh
                    }
                }
                return false
            });
            V.get(0).onselectstart = function() {
                return false
            };
            a0.click(function() {
                kw.val(V.html());
                aP.check();
                aQ("")
            })
        }
        V.text(bf);
        if (bf != "") {
            var bg = kw.textWidth();
            V.css({
                "margin-left": bg + 10 + "px",
                "max-width": a0.parent().width() - bg - 14 + "px"
            }).text(bf);
            a0.show()
        } else {
            a0.hide()
        }
        a0.hover(function() {
            $(this).css("opacity", 1)
        }, function() {
            $(this).css("opacity", 0.5)
        })
    }
    function aQ(bf) {
        if (window.__sam_tip) {
            return a(bf)
        }
        if (u) {
            if (V) {
                V.html("")
            }
            return 
        }
        if (pageState == 0) {
            return 
        }
        if (window.__disable_kw_tip) {
            return 
        }
        if (!V) {
            V = $('<div id="kw_tip" style="width:initial" unselectable="on" onselectstart="return false;" class="s_ipt_tip"/>').insertBefore(kw);
            V.parent().click(function(bk) {
                var bj = kw.get(0);
                if (bk.target === bj) {
                    return true
                }
                bj.focus();
                var bh = bj.value.length;
                if (document.selection) {
                    var bi = bj.createTextRange();
                    bi.moveStart("character", bh);
                    bi.collapse();
                    bi.select()
                } else {
                    if (typeof bj.selectionStart == "number" && typeof bj.selectionEnd == "number") {
                        bj.selectionStart = bj.selectionEnd = bh
                    }
                }
                return false
            });
            V.get(0).onselectstart = function() {
                return false
            }
        }
        V.text(bf);
        if (bf != "") {
            var bg = kw.textWidth();
            V.css({
                "margin-left": bg + 10 + "px",
                "max-width": V.parent().width() - bg - 14 + "px"
            }).text(bf).show()
        } else {
            V.hide()
        }
    }
    var u = false;
    function ar() {
        u = false
    }
    function b(bf) {
        u = true;
        if (window.__disable_swap_to_empty) {
            if (ao && ao.real_wd && $.trim(kw.val())) {
                aQ(ao.real_wd);
                I(ao)
            } else {
                aQ("");
                I()
            }
            return 
        }
        clearTimeout(U);
        U = false;
        aa.hide();
        if (pageState == 0) {
            an(bf)
        }
        ah = ao;
        ao = null;
        W = null;
        a1();
        if (bds && bds.se && bds.se.certification && bds.se.certification.data) {
            bds.se.certification.data = []
        }
        a8(function() {
            ay()
        });
        bds.clearReady();
        if ($.trim(kw.val())) {
            aa.html("<div style='margin:30px 0 0 162px;font-size:13px;color:#666'>请按“回车”键发起检索</div>")
        } else {
            aa.empty()
        }
        document.title = "百度搜索";
        aQ("");
        aa.show()
    }
    function I(bf) {
        var bg = aT(kw.val());
        if (bf && bg == bf.real_wd) {
            $("#super_se_tip").remove()
        }
    }
    $(window).on("swap_dom_ready", function(bh, bf) {
        var bg = "";
        if (bf && bf.real_wd&&!bf.no_predict) {
            bg = bf.real_wd
        }
        aQ(bg);
        I(bf)
    });
    $(window).on("swap_end", function(bg, bf) {
        if (!bf) {
            return 
        }
        if (U) {
            clearTimeout(U);
            U = false;
            az = null
        }
        bf.confirm = false;
        if (!bf.force) {
            az = r(bf, ah, 21);
            U = setTimeout(az, aN)
        } else {
            r(bf, ah, 20)()
        }
    });
    function aK(bi, bf) {
        L = {};
        var bk = new Date().getTime();
        if (!bf.force) {
            aE(bf, {
                utime: new Date().getTime() - aR
            })
        }
        if (!bf ||!bf.loaded) {
            return false
        }
        if (typeof bf.html == "string") {
            bf.html = $(bf.html)
        }
        $(bf).trigger("swap_begin");
        a8(function() {
            bf.pdc.mark("pt");
            $(window).trigger("swap_begin", [bf, bi]);
            var bn = aA && aA.getData();
            if (bn) {
                if (window.ispeed === true) {
                    aA.monitor(bn)
                }
                bf.pdc.setParam("upm", bn.join(","))
            }
        });
        a8(function() {
            bf.base64.restart();
            try {
                if (!bf.base64_loaded) {
                    var bo = $.parseJSON(bf.html.find("#img_list").text());
                    bf.base64.loadImg(bo.right, bo.left)
                }
            } catch (bn) {}
            bf.base64.end()
        });
        var bh = [$(window).scrollLeft(), $(window).scrollTop()];
        aa.hide();
        oldEnv = W;
        W = bi;
        ah = ao;
        ao = bf;
        bds.comm.cur_disp_query = bi.p("wd");
        a1();
        if (bds && bds.se && bds.se.certification && bds.se.certification.data) {
            bds.se.certification.data = []
        }
        if (pageState == 0) {
            an(bi)
        }
        a8(function() {
            ay()
        });
        bds.clearReady();
        aa.empty();
        var bj = bf.html;
        if (ag.use_cache_repeatedly) {
            bj = bj.clone()
        }
        a8(function() {
            bj.find("#head_style").children().removeAttr("data-for").appendTo("head")
        });
        a8(function() {
            $.globalEval(bj.find("#head_script").html())
        });
        a8(function() {
            bj.find("#content_script script").each(function(bn, bo) {
                $.globalEval($(bo).html())
            })
        });
        a8(function() {
            var bn = bj.find("#s_tab");
            if (!bn.size()) {
                return 
            }
            var bo = $("#s_tab");
            if (bo.size()) {
                bo.replaceWith(bn)
            } else {
                bn.insertBefore(aa)
            }
        });
        var bl = false;
        (function() {
            var bp = bj.find("#con-at");
            var bn = $("#con-at");
            var bq = bn.children().children();
            if (!bq.size()) {
                if (bp.children().size()) {
                    H(aa, bp, "insertBefore")
                }
            } else {
                if (!bp.children().size()) {
                    bn.remove();
                    $(window).trigger("top_result_removed")
                } else {
                    var bo = bp.children().children();
                    if (bq.attr("cq") != bo.attr("cq") || bq.attr("srcid") != bo.attr("srcid") || (bf.force && oldEnv && oldEnv.equals(W)) || (!W.p("cq") ||!W.p("srcid"))) {
                        bn.remove();
                        $(window).trigger("top_result_removed");
                        H(aa, bp, "insertBefore")
                    } else {
                        bl = true
                    }
                }
            }
        })();
        var bg = bj.find("#container");
        bf.pdc.bindImgLoad(bg);
        H(aa, bg);
        if (!$("#footer").size()) {
            var bm = bj.find("#footer").children();
            H(aa, bm)
        }
        a8(function() {
            var bn = new Date().getTime();
            if (bj) {
                $.globalEval(bj.find("#jsMerge").html())
            }
            aE(bf, {
                jsmergetime: new Date().getTime() - bn
            })
        });
        if (bds && bds.comm && bds.comm.templateName == bds.comm.resTemplateName) {
            if (bds.comm.seinfo) {
                bds.comm.seinfo.rsv_tpfail = 0
            }
        } else {
            if (bds.comm.seinfo) {
                bds.comm.seinfo.rsv_tpfail = 1
            }
        }
        if (pageState != 0 && bds && bds.util && bds.util.setContainerWidth) {
            bds.util.setContainerWidth()
        }
        document.title = bi.p("wd") + "_百度搜索";
        aa.show();
        aH();
        aE(bf, {
            domtime: new Date().getTime() - bk
        });
        aE(bf, {
            waittime: new Date().getTime() - aL
        });
        bf.pdc.mark("dom");
        $(window).trigger("swap_dom_ready", [bf, bi]);
        if (window.__lazy_foot_js) {
            setTimeout(function() {
                S(bi, bf, bk)
            }, 0)
        } else {
            S(bi, bf, bk)
        }
        if (!bl) {
            window.scrollTo(0, 0)
        } else {
            window.scrollTo(bh[0], bh[1])
        }
        $(window).trigger("scroll");
        swap_wait = false
    }
    function S(bg, bf, bi) {
        var bh;
        if (!bi) {
            bi = 0
        }
        if (bf) {
            bh = bf.html
        }
        a8(function() {
            a9.get(0).f.value = 8
        });
        a8(function() {
            var bj = new Date().getTime();
            if (bf && bf.base64) {
                bf.base64.setDomLoad("left");
                bf.base64.setDomLoad("right")
            }
            aE(bf, {
                base64time: new Date().getTime() - bj
            })
        });
        $("#search").find("form").submit(function() {
            var bk = kw;
            kw = $(this).find("[name='wd']");
            var bj = ax.call(this);
            kw = bk;
            return bj
        });
        a8(function() {
            var bj = new Date().getTime();
            bds.doReady();
            aE(bf, {
                bdstime: new Date().getTime() - bj
            })
        });
        a8(function() {
            var bj = new Date().getTime();
            if (bh) {
                $.globalEval(bh.find("#ecomScript").html())
            }
            aE(bf, {
                ecomtime: new Date().getTime() - bj
            })
        });
        a8(function() {
            var bj = new Date().getTime();
            if (bds.se.tools) {
                if (O) {
                    clearTimeout(O)
                }
                O = setTimeout(function() {
                    bds.se.tools()
                }, 600)
            }
            if (bds && bds.se && bds.se.certification && bds.se.certification.build) {
                if (ad) {
                    clearTimeout(ad)
                }
                ad = setTimeout(function() {
                    if ($(".certification").size() > 0) {
                        bds.se.certification.build.init()
                    }
                }, 1000)
            }
            if (bds && bds.se && bds.se.safeTip) {
                if (T) {
                    clearTimeout(T)
                }
                T = setTimeout(function() {
                    if ($(".unsafe_ico_new").size() > 0) {
                        bds.se.safeTip.init()
                    }
                }, 1200)
            }
            aE(bf, {
                tiptime: new Date().getTime() - bj
            })
        });
        a8(function() {
            var bj = new Date().getTime();
            if (aO) {
                clearTimeout(aO)
            }
            aO = setTimeout(function() {
                if (bds.se.login && bds.se.login.setUserInfo) {
                    bds.se.login.setUserInfo()
                }
                if (bds.su && bds.su.U && bds.su.U.urTrigger.init) {
                    bds.su.U.urTrigger.init()
                } else {
                    if ((bds.comm.newindex || bds.comm.username) && bds.su.urStatic) {
                        var bk = $.ajax({
                            dataType: "script",
                            cache: true,
                            url: bds.su.urStatic + "/static/ur/js/ur/urstatic-async-1.4.js"
                        });
                        bk.done(function() {
                            bds.su.U.urTrigger.init()
                        })
                    }
                }
            }, 1400);
            aE(bf, {
                urtime: new Date().getTime() - bj
            })
        });
        a8(function() {
            var bj = new Date().getTime();
            window.initResultClickLog();
            aE(bf, {
                clicktime: new Date().getTime() - bj
            })
        });
        a8(function() {
            aE(bf, {
                rtime: new Date().getTime() - bi,
                used: 1
            });
            if (bds.comm.seinfo && bf) {
                bds.comm.seinfo.rsv_pstg = bf.type
            }
        });
        typeof a7 == "function" ? a7() : "";
        a8(function() {
            if (!bf) {
                return 
            }
            var bk = "<table>";
            for (var bj in bf.log) {
                if (bf.log.hasOwnProperty(bj)) {
                    bk += "<tr><td>" + bj + "</td><td>" + encodeURIComponent(bf.log[bj]) + "</td></tr>"
                }
            }
            for (var bj in L) {
                if (L.hasOwnProperty(bj)) {
                    bk += "<tr><td>" + bj + "</td><td>" + encodeURIComponent(L[bj]) + "</td></tr>"
                }
            }
            bk += "</table>";
            $("#debug").html(bk)
        });
        a8(function() {
            $(window).trigger("swap_end", [bf, bg]);
            ba();
            aR = new Date().getTime();
            if (bf && bf.pdc) {
                bf.pdc.mark("js");
                bf.pdc.trigger("swap_end")
            }
        })
    }
    function a1() {
        a8(function() {
            $.each(bds.comm.tips, function(bf, bg) {
                if (bg && bg.destroy) {
                    bg.destroy()
                }
            });
            $("#c-tips-container").empty();
            bds.comm.tips = []
        });
        a8(function() {
            if (window.app && window.app.dispose) {
                window.app.dispose()
            }
        });
        a8(function() {
            bds.comm.resolveUnloadHandler()
        });
        if (bds && bds.se && bds.se.certification && bds.se.certification.data) {
            bds.se.certification.data = []
        }
    }
    function ac() {
        if (U && az) {
            clearTimeout(U);
            U = setTimeout(az, aN)
        }
    }
    function r(bf, bh, bg) {
        return function() {
            if (!bf || bf.confirm) {
                return 
            }
            bds.comm.cur_query = bf.real_wd;
            bf.confirm = true;
            U = false;
            az = null;
            var bj = {};
            if (bh && bh.env) {
                bj.is_referer = bh.env.buildSyncSearchUrl()
            } else {
                bj.is_referer = aY
            }
            bj.is_xhr = "1";
            var bi = new P(bc(ak.getQueryString()));
            if (!bf.env.equals(bi)&&!bf.env.clone({
                wd: bf.prw
            }).equals(bi)) {
                ak.setUrl(bf.env)
            }
            if (!bf.seq) {
                bf.seq = 1
            } else {
                bf.seq++
            }
            if (bf.pdc) {
                if (bg != 20) {
                    bf.pdc.mark("st")
                }
                if (bf.pdc && bf.pdc.setParam) {
                    bf.pdc.setParam("cus_pstg", bg)
                }
                bf.pdc.trigger("confirm")
            }
            var bk = "/s?ie=utf-8&csq=" + bf.seq + "&pstg=" + bg + "&mod=2&isbd=1&cqid=" + bf.qid + "&istc=" + (new Date().getTime() - bf.startTime) + "&ver=" + bds.comm.baiduis_verify + "&chk=" + bf.chk + "&isid=" + au + "&" + bf.env.buildQueryString() + (bf.force ? "&f4s=1" : "") + (typeof s == "function" ? "&_ck=" + s(bf.env.p("wd")) : "");
            R.done(function() {
                if (aP.getStat("rsv_sug6")) {
                    bk += "&rsv_sug6=" + aP.getStat("rsv_sug6")
                }
            });
            $.ajax({
                headers: bj,
                url: bk
            });
            if (bds.comm.seinfo) {
                bds.comm.seinfo.rsv_prw = encodeURIComponent(kw.val());
                bds.comm.seinfo.rsv_pstg = bg
            }
            aJ();
            au = bf.qid;
            R.done(function() {
                if (bg == 20) {
                    aP.updateInitData()
                }
                aP.clearStat()
            })
        }
    }
    $(window).on("indexOff", function(bg, bf) {
        R.done(function() {
            aQ(bf.p("wd"))
        })
    });
    if (ak.support() && al.attr("target") != "_blank") {
        R.done(function() {
            aP.setMaxNum(4)
        })
    }
    var aq = false, M;
    var aU = false;
    a9.mousedown(function() {
        aU = false
    }).delegate("a,input", "focus", function() {
        aU = false
    });
    function ax(bg) {
        if (!ak.support()) {
            return true
        }
        if (aU) {
            return false
        }
        aU = true;
        kw.blur();
        aQ("");
        if ($(this).attr("target")) {
            return true
        }
        aq = true;
        M = setTimeout(function() {
            aq = false
        }, 1000);
        try {
            if (!$.trim(kw.val())) {
                a5.href = "/";
                return false
            }
            var bj = new P();
            var bi = $(this).serializeArray(), bf;
            for (bf = 0; bf < bi.length; bf++) {
                bj.p(bi[bf].name, bi[bf].value)
            }
            bj.p("wd", kw.val());
            if (bj.p("nojc")) {
                bj.p("nojc", "")
            }
            if (ao) {
                if ((bj.equals(ao.env.clone({
                    wd: ao.real_wd
                })))&&!ao.force) {
                    r(ao, ah, 22)();
                    ao.force = true;
                    aQ("");
                    I(ao);
                    return false
                }
                if (ao.env.p("rsv_spt")) {
                    bj.p("rsv_spt", ao.env.p("rsv_spt"))
                }
            }
            ak.submit(bj, !!bg)
        } catch (bh) {}
        return false
    }
    var aF = {};
    function q(bh) {
        var bi = {
            force: false,
            no_predict: false,
            use_cache: false,
            shouldShow: true
        };
        if (bh) {
            $.extend(bi, bh)
        }
        var bk = bi.env, bm = bi.force, bl = bi.no_predict, bg = bi.shouldShow, bj = bi.use_cache, bf;
        if (!bk ||!bk.p("wd") ||!bk.hashCode()) {
            return 
        }
        if ((am || Cookie.get("ISSW") == 1)&&!bm && bl) {
            return
        }
        if ((y || Cookie.get("ISSW") == 1)&&!bm&&!bl) {
            return 
        }
        if (bj && (bf = i.hasCache(bk, {
            loaded: true
        }))) {
            if (bg) {
                if (!ao ||!bf.env.clone({
                    wd: bf.real_wd
                }).equals(ao.env.clone({
                    wd: ao.real_wd
                }))) {
                    bf.force = bi.force;
                    bf.no_predict = bi.no_predict;
                    bf.pdc.init();
                    if (bf.force) {
                        bf.pdc.mark("st")
                    }
                    aK(bk, bf)
                }
                aQ(bi.no_predict ? "" : bf.real_wd);
                I(bf)
            }
            return 
        }
        if (bm && bg && bl) {
            p()
        }
        bf = {
            env: bk,
            no_predict: bl,
            shouldShow: bg,
            loaded: false,
            force: bm,
            startTime: new Date().getTime(),
            log: {
                ctime: new Date().getTime() - aR,
                wd: bk.p("wd"),
                ntime: 0,
                stat: 0,
                used: 0,
                rtime: 0,
                utime: (bm ? new Date().getTime() - aR : 0),
                res: 0
            }
        };
        bf.pdc = aw(bf);
        if (bf.force) {
            bf.pdc.mark("st")
        }
        bf.base64 = isbase64(bf.pdc);
        aZ.push(bf.log);
        x++;
        d(bf)
    }
    function aD() {
        var bf = [];
        if (af) {
            bf = $.map(af.slice(0, 2), function(bg) {
                return bg.value
            })
        }
        return bf.join("\t")
    }
    function bd(bf) {
        D();
        a5.replace(bf.buildSyncSearchUrl())
    }
    var B, E;
    function ab(bg, bf) {
        if (!bf) {
            am = true;
            if (B) {
                clearTimeout(B);
                B = false
            }
            B = setTimeout(function() {
                am = N
            }, bg)
        } else {
            y = true;
            if (E) {
                clearTimeout(E);
                E = false
            }
            E = setTimeout(function() {
                y = a4
            }, bg)
        }
    }
    function d(bm) {
        var bl;
        var bn = bm.env;
        var bj = {};
        var bf;
        if (ao && ao.env) {
            bj.is_referer = ao.env.buildSyncSearchUrl()
        } else {
            bj.is_referer = aY
        }
        bj.is_xhr = "1";
        if (window.bds && bds.comm && bds.comm.cur_query) {
            bn.p("bs", bds.comm.cur_query)
        } else {
            bn.p("bs", "")
        }
        if (window.bds && bds.comm && bds.comm.cur_disp_query) {
            bj.is_pbs = encodeURIComponent(bds.comm.cur_disp_query)
        }
        var bg = "ie=utf-8" + (bds.comm.newindex ? "&newi=1" : "") + (aj.sid ? "&sid=" + encodeURIComponent(aj.sid) : "") + (aj.tnp ? "&tnp=" + encodeURIComponent(aj.tnp) : "") + "&mod=" + (bm.no_predict ? "1" : "11") + "&isbd=1&isid=" + au + "&" + bn.buildQueryString() + "&rsv_sid=" + bds.comm.indexSid + "&_ss=1&clist=" + encodeURIComponent(i.getCacheList()) + "&hsug=" + encodeURIComponent(aD()) + (bm.force ? "&f4s=1" : "") + "&csor=" + k(kw.get(0));
        var bh = "/s?" + bg;
        bh += "&_cr1=" + ap(bh);
        if (!bm.no_predict) {
            bf = i.find(function(bq) {
                if (!bq.loaded&&!bq.no_predict) {
                    return true
                }
            });
            for (bl = 0; bl < bf.length; bl++) {
                i.deleteCache(bf[bl])
            }
        }
        if (bm.no_predict&&!bm.force) {
            bf = i.find(function(bq) {
                if (bq.force && bn.equals(bq.env)) {
                    return true
                }
            });
            if (bf.length > 0) {
                return
            }
        }
        if (bm.force && bm.shouldShow) {
            var bo = false;
            bf = i.find(function(br) {
                var bq = bn.equals(br.env);
                if (!br.loaded && br.no_predict && br.force && bq && br !== bm) {
                    br.shouldShow = br.shouldShow || bm.shouldShow;
                    bo = true
                }
                if (!br.loaded&&!bq) {
                    return true
                } else {
                    br.pdc.mark("st")
                }
            });
            if (bo) {
                return 
            }
            for (bl = 0; bl < bf.length; bl++) {
                i.deleteCache(bf[bl])
            }
        }
        var bi = function(bs, br, bu) {
            if (br == 0) {
                bk(bs, bu);
                if (bm.pdc) {
                    bm.pdc.setParam("cus_srv", bds.se.mon.srvt)
                }
            } else {
                if (br == 1) {
                    try {
                        var bq = new Date() * 1;
                        bm.b64ildata = $.parseJSON(bs);
                        bm.base64.ilparseTime = new Date() * 1 - bq;
                        if (bm === ao) {
                            bm.base64.inline(bm.b64ildata);
                            bm.base64.ilrenderTime = new Date() * 1 - bq
                        }
                        $(bm).one("swap_begin", function() {
                            this.base64.inline(this.b64ildata, this.html.get(0))
                        })
                    } catch (bt) {}
                } else {
                    if (br == 2) {
                        if (bm.base64) {
                            bm.base64.ilsum = bs
                        }
                    }
                }
            }
        };
        var bk = function(bw, bG) {
            if ((bG && bG.status == "302") || (bw && bw.indexOf("<div>") > 10)) {
                if (bm.force) {
                    bd(bn)
                } else {
                    i.deleteCache(bm)
                }
                return 
            }
            aE(bm, {
                ntime: new Date().getTime() - bm.startTime,
                res: 1
            });
            var bC;
            var br = "<!--data-->";
            var bu = bw.indexOf(br);
            if (bu!=-1) {
                bC = $(bw.substr(0, bu));
                bm.html = bw.substr(bu + br.length);
                try {
                    var bt = $.parseJSON(bC.find("#img_list").text());
                    bm.base64.loadImg(bt.right, bt.left);
                    bm.base64_loaded = true
                } catch (by) {}
                try {
                    aX(bC.find("#limg_list").text())
                } catch (by) {}
            } else {
                bC = bm.html = $(bw)
            }
            var bF = parseInt(bC.find("#__status").eq(0).html());
            var bD = parseInt(bC.find("#__switchtime").eq(0).html());
            var bx = parseInt(bC.find("#__redirect").eq(0).html());
            bm.real_wd = bC.find("#__real_wd").eq(0).text();
            bm.real_wd_org = bC.find("#__real_wd_org").eq(0).text();
            var bs = false;
            if (bn.p("wd") == aT(kw.val()) && bm.shouldShow) {
                bs = true
            }
            if (bm.real_wd) {
                bm.prw = bn.p("wd");
                bn.p("wd", bm.real_wd)
            }
            var bB = bC.find("#__queryId").html();
            var bE = bC.find("#__querySign").html();
            bm.querySign = bE;
            aE(bm, {
                stat: (bF ? bF : 0)
            });
            if (bds.comm.isDebug) {
                $("#isDebugInfo").html(bC.find("#__isDebugInfo").html())
            }
            if (bB) {
                bm.qid = bB
            }
            var bA = bC.find("#__chk").html();
            bm.chk = bA ? bA : 0;
            if (!bw || (!bB&&!bD&&!bx&&!bF) || (!bE && bm.force)) {
                if (bm.force) {
                    bn.p("__eis", 1);
                    bn.p("__eist", bw ? bw.length : 0);
                    bn.p("real_wd", bm.real_wd);
                    bd(bn);
                    return
                } else {
                    i.deleteCache(bm);
                    return 
                }
            }
            if (bD > 0) {
                ab(bD * 1000, !bm.no_predict)
            }
            if (bF==-11) {
                var bz = i.getCacheBySign(bE);
                if (!bz) {
                    q({
                        env: bm.env.clone({
                            wd: bm.real_wd
                        }),
                        force: bm.force,
                        use_cache: false,
                        no_predict: true
                    });
                    i.deleteCache(bm);
                    return 
                }
                bz.force = bm.force;
                ar();
                aQ(bz.real_wd);
                I(bz);
                i.deleteCache(bm);
                bm = bz;
                if (!ao || bm.real_wd != ao.real_wd) {
                    bs = true
                }
            } else {
                if (bF < 0) {
                    if (bx == 1 && bm.force) {
                        aE(bm, {
                            redirect: 1
                        });
                        bd(bn);
                        return 
                    }
                    b();
                    i.deleteCache(bm);
                    return 
                } else {
                    if (bF > 0) {
                        i.deleteCache(bm);
                        return 
                    }
                }
            }
            var bq = i.find(function(bH) {
                if (!bH.loaded && bH !== bm && bH.no_predict && bn.equals(bH.env)) {
                    if (bH.shouldShow) {
                        bs = true
                    }
                    if (bH.force) {
                        bm.force = true
                    }
                    return true
                }
            });
            for (var bv = 0; bv < bq.length; bv++) {
                i.deleteCache(bq[bv])
            }
            bm.loaded = true;
            if ((i.shouldShow(bm) || bs) && bm !== ao || bm.force) {
                ar();
                bm.shouldShow = false;
                if (bF==-11) {
                    bm.pdc.init()
                } else {
                    bm.pdc.mark("net")
                }
                aK(bn, bm)
            }
        };
        var bp;
        bp = $.ajax({
            dataType: "parts",
            url: bh,
            headers: bj,
            delimiter: "</*3*/>"
        });
        bp.parts(function(br, bq, bs) {
            bi(br, bq, bp)
        });
        bm.xhr = bp;
        i.addCache(bm)
    }
    function ba() {
        var bf;
        aZ = [];
        x = 0;
        aq = false;
        clearTimeout(M)
    }
    $(window).on("swap_end", function(bg, bf) {
        if (!bf) {
            aJ()
        }
    });
    function o() {
        if (window.index_off) {
            window.index_off()
        }
        if (index_kw[0] !== a6[0]) {
            index_kw.val("")
        }
        kw = a6;
        pageState = 1;
        bds.comm.ishome = 0;
        bds.comm.cur_query = bds.comm.query;
        W = new P();
        ao = {
            env: W,
            real_wd: bds.comm.query,
            force: true,
            confirm: true
        };
        R.done((function(bf) {
            return function() {
                j.start()
            }
        })());
        $(window).trigger("index_off");
        bds.util.setContainerWidth();
        a8(function() {
            $(window).trigger("swap_dom_ready")
        });
        if (window.__lazy_foot_js) {
            setTimeout(function() {
                S()
            }, 0)
        } else {
            S()
        }
    }
    function an(bf) {
        if (window.index_off) {
            window.index_off()
        }
        if (index_kw.get(0) !== a6.get(0)) {
            index_kw.val("");
            a6.val(bf.p("wd"))
        }
        kw = a6;
        pageState = 1;
        bds.comm.ishome = 0;
        R.done(function() {
            if (aC !== j) {
                aC.stop();
                j.hide();
                j.setKey(bf.p("wd"));
                j.start()
            }
        });
        bds.util.setContainerWidth();
        $(window).trigger("index_off", bf)
    }
    if (window.__click_input_swap) {
        kw.click(function() {
            if (pageState == 0) {
                var bf = new P({
                    pn: "",
                    wd: kw.val()
                });
                an(bf)
            }
        })
    }
    ak.init();
    $(function() {
        var bf = $("script").last();
        var bg = $("head");
        ay = function() {
            bf.nextAll().not("[data-for]").not("#passport-login-pop").remove();
            bg.find("*").not("[data-for]").not("meta").not("title").not("script[async]").not('link[href*="passport"]').remove()
        }
    });
    if (bds.comm.resultPage) {
        o()
    }
    t.delegate("#s_tab a", "mousedown", function() {
        setHeadUrl(this)
    }).delegate("#s_tab a", "focusin", function() {
        setHeadUrl(this)
    });
    aa.delegate("#page strong+a,#page a.n", "mouseover", function() {
        q({
            env: new P(bc($(this).attr("href"))),
            force: false,
            use_cache: true,
            no_predict: true,
            shouldShow: false
        })
    });
    var Z, X;
    function aW() {
        Z = false;
        X = false
    }
    function m(bh) {
        if (!Z) {
            Z = {
                x: bh.pageX,
                y: bh.pageY
            }
        }
        if (Math.abs(Z.x - bh.pageX) > f || Math.abs(Z.y - bh.pageY) > f) {
            if (!X) {
                var bf = aP;
                var bg = window.__preload_more ? 2: 1;
                if (bg && bf && bf.data() && bf.data()[0] && bf.visible()) {
                    q({
                        env: new P().clone({
                            wd: bf.data()[0].value
                        }),
                        force: false,
                        no_predict: true,
                        use_cache: true,
                        shouldShow: false
                    });
                    bg--
                }
                if (bg && bf && bf.data() && bf.data()[1] && bf.visible()) {
                    q({
                        env: new P().clone({
                            wd: bf.data()[1].value
                        }),
                        force: false,
                        no_predict: true,
                        use_cache: true,
                        shouldShow: false
                    });
                    bg--
                }
                if (window.__preload_more && bg && $.trim(kw.val()) && (!ao || ao.env.p("wd") != $.trim(kw.val()))) {
                    q({
                        env: new P().clone({
                            wd: $.trim(kw.val())
                        }),
                        force: false,
                        no_predict: true,
                        use_cache: true,
                        shouldShow: false
                    });
                    bg--
                }
            }
            X = true
        }
    }
    R.done(function() {
        $(document).mousemove(m)
    });
    $("#u .back_org").click(function() {
        var bf = new Date();
        bf.setTime(new Date().getTime() + 1103760000000);
        Cookie.set("ORIGIN", 2, document.domain, "/", bf);
        if (W) {
            a5.replace(W.buildSyncSearchUrl({
                _r: Math.random()
            }))
        } else {
            a5.href = "/"
        }
    });
    $(window).scroll((function() {
        var bh = $("#head"), bf = $(window);
        var bg = 40;
        var bk;
        var bj = bh.offset().top;
        var bi = function() {
            if (bk) {
                clearTimeout(bk);
                bk = false
            }
            bk = setTimeout(function() {
                var bl = bf.scrollTop();
                if (bl > bg + bj) {
                    bk = setTimeout(function() {
                        bh.addClass("s_down");
                        R.done(function() {
                            j.hide()
                        })
                    }, 0)
                } else {
                    if (bl <= bg + bj) {
                        bk = setTimeout(function() {
                            bh.removeClass("s_down")
                        }, 0)
                    }
                }
            }, 50)
        };
        bi();
        return bi
    })());
    kw.bind("paste", function(bg) {
        if (window.__disable_index_predict && pageState == 0) {
            return 
        }
        var bh = this;
        var bf = this.value;
        J = true;
        setTimeout(function() {
            if (bh.value && bh.value != bf) {
                q({
                    env: new P().clone({
                        wd: $.trim(bh.value)
                    }),
                    force: false,
                    use_cache: true,
                    no_predict: true,
                    shouldShow: true
                })
            }
        }, 0)
    })
}
var define;
var require;
var esl;
(function(b) {
    var o = {};
    var C = {};
    var x = 1;
    var g = 2;
    var s = 3;
    var p = 4;
    var ah = X();
    var N;
    function i(ap, aq) {
        var an = [];
        function am(ar) {
            if (ar.indexOf(".") === 0) {
                an.push(ar)
            }
        }
        if (typeof ap === "string") {
            am(ap)
        } else {
            aa(ap, function(ar) {
                am(ar)
            })
        }
        if (an.length > 0) {
            throw new Error("[REQUIRE_FATAL]Relative ID is not allowed in global require: " + an.join(", "))
        }
        var ao = n.waitSeconds;
        if (ao && (ap instanceof Array)) {
            if (N) {
                clearTimeout(N)
            }
            N = setTimeout(I, ao * 1000)
        }
        return ah(ap, aq)
    }
    i.version = "1.8.6";
    i.loader = "esl";
    i.toUrl = ah.toUrl;
    function I() {
        var aq = [];
        var ar = [];
        var ao = {};
        var ap = {};
        var an = {};
        function am(aw, av) {
            if (an[aw] || l(aw, p)) {
                return 
            }
            an[aw] = 1;
            if (!l(aw, s)) {
                if (!ao[aw]) {
                    ao[aw] = 1;
                    aq.push(aw)
                }
            }
            var au = o[aw];
            if (!au) {
                if (!ap[aw]) {
                    ap[aw] = 1;
                    ar.push(aw)
                }
            } else {
                if (av) {
                    if (!ao[aw]) {
                        ao[aw] = 1;
                        aq.push(aw)
                    }
                    aa(au.depMs, function(ax) {
                        am(ax.absId, ax.hard)
                    })
                }
            }
        }
        for (var at in C) {
            am(at, 1)
        }
        if (aq.length || ar.length) {
            throw new Error("[MODULE_TIMEOUT]Hang( " + (aq.join(", ") || "none") + " ) Miss( " + (ar.join(", ") || "none") + " )")
        }
    }
    var z;
    function R(aq, ap, ao) {
        if (ao == null) {
            if (ap == null) {
                ao = aq;
                aq = null
            } else {
                ao = ap;
                ap = null;
                if (aq instanceof Array) {
                    ap = aq;
                    aq = null
                }
            }
        }
        if (ao == null) {
            return 
        }
        var am = window.opera;
        if (!aq && document.attachEvent && (!(am && am.toString() === "[object Opera]"))) {
            var an = t();
            aq = an && an.getAttribute("data-require-id")
        }
        if (aq) {
            y(aq, ap, ao);
            if (z) {
                clearTimeout(z)
            }
        } else {
            r[0] = {
                deps: ap,
                factory: ao
            }
        }
    }
    R.amd = {};
    function ad() {
        var am = n.config[this.id];
        if (am && typeof am === "object") {
            return am
        }
        return {}
    }
    function y(ao, an, am) {
        if (!o[ao]) {
            o[ao] = {
                id: ao,
                depsDec: an,
                deps: an || ["require", "exports", "module"],
                factoryDeps: [],
                factory: am,
                exports: {},
                config: ad,
                state: x,
                require: X(ao),
                depMs: [],
                depMkv: {},
                depRs: [],
                depPMs: []
            }
        }
    }
    function ag(ar) {
        var ao = o[ar];
        if (!ao || l(ar, g)) {
            return 
        }
        var aq = ao.deps;
        var an = ao.factory;
        var ap = 0;
        if (typeof an === "function") {
            ap = Math.min(an.length, aq.length);
            !ao.depsDec && an.toString().replace(/(\/\*([\s\S]*?)\*\/|([^:]|^)\/\/(.*)$)/mg, "").replace(/require\(\s*(['"'])([^'"]+)\1\s*\)/g, function(au, at, av) {
                aq.push(av)
            })
        }
        var am = [];
        aa(aq, function(au, at) {
            var ax = W(au);
            var aw = af(ax.mod, ar);
            var ay;
            var av;
            if (aw&&!B[aw]) {
                if (ax.res) {
                    av = {
                        id: au,
                        mod: aw,
                        res: ax.res
                    };
                    C[aw] = 1;
                    ao.depPMs.push(aw);
                    ao.depRs.push(av)
                }
                ay = ao.depMkv[aw];
                if (!ay) {
                    ay = {
                        id: ax.mod,
                        absId: aw,
                        hard: at < ap
                    };
                    ao.depMs.push(ay);
                    ao.depMkv[aw] = ay;
                    am.push(aw)
                }
            } else {
                ay = {
                    absId: aw
                }
            }
            if (at < ap) {
                ao.factoryDeps.push(av || ay)
            }
        });
        ao.state = g;
        O(ar);
        a(am)
    }
    function S() {
        for (var am in C) {
            E(am);
            al(am)
        }
    }
    function E(ao) {
        var am = {};
        an(ao);
        function an(ar) {
            if (!l(ar, g)) {
                return false
            }
            if (l(ar, s) || am[ar]) {
                return true
            }
            am[ar] = 1;
            var ap = o[ar];
            var aq = true;
            aa(ap.depMs, function(at) {
                return (aq = an(at.absId))
            });
            aq && aa(ap.depRs, function(at) {
                aq=!!(at.absId && l(at.absId, p));
                return aq
            });
            if (aq) {
                ap.state = s
            }
            return aq
        }
    }
    function O(ap) {
        var an = o[ap];
        var ao;
        an.invokeFactory = am;
        aa(an.depPMs, function(aq) {
            f(aq, function() {
                aa(an.depRs, function(ar) {
                    if (!ar.absId && ar.mod === aq) {
                        ar.absId = af(ar.id, ap);
                        a([ar.absId], S)
                    }
                })
            })
        });
        function am() {
            if (ao || an.state !== s) {
                return 
            }
            ao = 1;
            var aw = 1;
            var ax = [];
            aa(an.factoryDeps, function(az) {
                var ay = az.absId;
                if (!B[ay]) {
                    al(ay);
                    if (!l(ay, p)) {
                        aw = 0;
                        return false
                    }
                }
                ax.push(ay)
            });
            if (aw) {
                try {
                    var au = ae(ax, {
                        require: an.require,
                        exports: an.exports,
                        module: an
                    });
                    var at = an.factory;
                    var ar = typeof at === "function" ? at.apply(b, au): at;
                    if (ar != null) {
                        an.exports = ar
                    }
                    an.invokeFactory = null;
                    delete C[ap]
                } catch (av) {
                    ao = 0;
                    if (/^\[MODULE_MISS\]"([^"]+)/.test(av.message)) {
                        var aq = an.depMkv[RegExp.$1];
                        aq && (aq.hard = 1);
                        return 
                    }
                    throw av
                }
                D(ap)
            }
        }
    }
    function l(an, am) {
        return o[an] && o[an].state >= am
    }
    function al(an) {
        var am = o[an];
        if (am && am.invokeFactory) {
            am.invokeFactory()
        }
    }
    function ae(an, ao) {
        var am = [];
        aa(an, function(aq, ap) {
            am[ap] = ao[aq] || u(aq)
        });
        return am
    }
    var V = {};
    function f(ao, an) {
        if (l(ao, p)) {
            an();
            return 
        }
        var am = V[ao];
        if (!am) {
            am = V[ao] = []
        }
        am.push(an)
    }
    function D(ap) {
        var ao = V[ap] || [];
        var an = o[ap];
        an.state = p;
        var am = ao.length;
        while (am--) {
            ao[am]()
        }
        ao.length = 0;
        delete V[ap]
    }
    function u(am) {
        if (l(am, p)) {
            return o[am].exports
        }
        return null
    }
    var B = {
        require: i,
        exports: 1,
        module: 1
    };
    var r = [];
    function q(am) {
        aa(r, function(an) {
            y(am, an.deps, an.factory)
        });
        r.length = 0;
        ag(am)
    }
    function a(ap, ar, am, an) {
        if (typeof ap === "string") {
            al(ap);
            if (!l(ap, p)) {
                throw new Error('[MODULE_MISS]"' + ap + '" is not exists!')
            }
            return u(ap)
        }
        an = an || {};
        var aq = 0;
        if (ap instanceof Array) {
            ao();
            if (!aq) {
                aa(ap, function(at) {
                    if (!(B[at] || l(at, p))) {
                        f(at, ao);
                        if (!an[at]) {
                            (at.indexOf("!") > 0 ? F : j)(at, am)
                        }
                        ag(at)
                    }
                });
                S()
            }
        }
        function ao() {
            if (!aq) {
                var at = 1;
                aa(ap, function(au) {
                    if (!B[au]) {
                        return (at=!!l(au, p))
                    }
                });
                if (at) {
                    aq = 1;
                    (typeof ar === "function") && ar.apply(b, ae(ap, B))
                }
            }
        }
    }
    var P = {};
    function j(an) {
        if (P[an] || o[an]) {
            return 
        }
        P[an] = 1;
        var am = document.createElement("script");
        am.setAttribute("data-require-id", an);
        am.src = K(an + ".js");
        am.async = true;
        if (am.readyState) {
            am.onreadystatechange = ao
        } else {
            am.onload = ao
        }
        ak(am);
        function ao() {
            var ap = am.readyState;
            if (typeof ap === "undefined" || /^(loaded|complete)$/.test(ap)) {
                am.onload = am.onreadystatechange = null;
                am = null;
                q(an);
                for (var aq in C) {
                    ag(aq)
                }
                S()
            }
        }
    }
    function F(am, ao) {
        if (o[am]) {
            return 
        }
        var ar = W(am);
        var aq = {
            id: am,
            state: g
        };
        o[am] = aq;
        function an(at) {
            aq.exports = at || true;
            D(am)
        }
        an.fromText = function(au, at) {
            C[au] = 1;
            new Function(at)();
            q(au)
        };
        function ap(at) {
            var au = ao ? o[ao].require: ah;
            at.load(ar.res, au, an, ad.call({
                id: am
            }))
        }
        ap(u(ar.mod))
    }
    var n = {
        baseUrl: "./",
        paths: {},
        config: {},
        map: {},
        packages: [],
        waitSeconds: 0,
        noRequests: {},
        urlArgs: {}
    };
    i.config = function(an) {
        function ap(ar) {
            am.push(ar)
        }
        if (an) {
            for (var ao in n) {
                var aq = an[ao];
                var am = n[ao];
                if (aq) {
                    if (ao === "urlArgs" && typeof aq === "string") {
                        n.urlArgs["*"] = aq
                    } else {
                        if (am instanceof Array) {
                            aa(aq, ap)
                        } else {
                            if (typeof am === "object") {
                                for (var ao in aq) {
                                    am[ao] = aq[ao]
                                }
                            } else {
                                n[ao] = aq
                            }
                        }
                    }
                }
            }
            aj()
        }
    };
    aj();
    var L;
    var J;
    var M;
    var ac;
    var ab;
    function H(ao, am) {
        var an = U(ao, 1, am);
        an.sort(m);
        return an
    }
    function aj() {
        n.baseUrl = n.baseUrl.replace(/\/$/, "") + "/";
        L = H(n.paths);
        M = H(n.map, 1);
        aa(M, function(am) {
            am.v = H(am.v)
        });
        J = [];
        aa(n.packages, function(am) {
            var an = am;
            if (typeof am === "string") {
                an = {
                    name: am.split("/")[0],
                    location: am,
                    main: "main"
                }
            }
            an.location = an.location || an.name;
            an.main = (an.main || "main").replace(/\.js$/i, "");
            an.reg = Y(an.name);
            J.push(an)
        });
        J.sort(m);
        ac = H(n.urlArgs, 1);
        ab = H(n.noRequests);
        aa(ab, function(an) {
            var ao = an.v;
            var am = {};
            an.v = am;
            if (!(ao instanceof Array)) {
                ao = [ao]
            }
            aa(ao, function(ap) {
                am[ap] = 1
            })
        })
    }
    function Z(an, am, ao) {
        aa(am, function(ap) {
            if (ap.reg.test(an)) {
                ao(ap.v, ap.k, ap);
                return false
            }
        })
    }
    function K(ap) {
        var au = /(\.[a-z0-9]+)$/i;
        var an = /(\?[^#]*)$/;
        var ar = "";
        var at = ap;
        var ao = "";
        if (an.test(ap)) {
            ao = RegExp.$1;
            ap = ap.replace(an, "")
        }
        if (au.test(ap)) {
            ar = RegExp.$1;
            at = ap.replace(au, "")
        }
        var am = at;
        var aq;
        Z(at, L, function(aw, av) {
            am = am.replace(av, aw);
            aq = 1
        });
        if (!aq) {
            Z(at, J, function(ax, av, aw) {
                am = am.replace(aw.name, aw.location)
            })
        }
        if (!/^([a-z]{2,10}:\/)?\//i.test(am)) {
            am = n.baseUrl + am
        }
        am += ar + ao;
        Z(at, ac, function(av) {
            am += (am.indexOf("?") > 0 ? "&" : "?") + av
        });
        return am
    }
    function X(am) {
        var ao = {};
        function an(ar, av) {
            if (typeof ar === "string") {
                if (!ao[ar]) {
                    ao[ar] = a(af(ar, am))
                }
                return ao[ar]
            } else {
                if (ar instanceof Array) {
                    var au = [];
                    var ap = [];
                    var at = [];
                    aa(ar, function(az, aw) {
                        var ay = W(az);
                        var ax = af(ay.mod, am);
                        ap.push(ax);
                        C[ax] = 1;
                        if (ay.res) {
                            au.push(ax);
                            at[aw] = null
                        } else {
                            at[aw] = ax
                        }
                    });
                    var aq = {};
                    aa(ap, function(ax) {
                        var aw;
                        Z(ax, ab, function(ay) {
                            aw = ay
                        });
                        if (aw) {
                            if (aw["*"]) {
                                aq[ax] = 1
                            } else {
                                aa(ap, function(ay) {
                                    if (aw[ay]) {
                                        aq[ax] = 1;
                                        return false
                                    }
                                })
                            }
                        }
                    });
                    a(ap, function() {
                        aa(at, function(ax, aw) {
                            if (ax == null) {
                                at[aw] = af(ar[aw], am)
                            }
                        });
                        a(at, av, am)
                    }, am, aq)
                }
            }
        }
        an.toUrl = function(ap) {
            return K(af(ap, am))
        };
        return an
    }
    function af(ar, am) {
        if (!ar) {
            return ""
        }
        am = am || "";
        var ap = W(ar);
        if (!ap) {
            return ar
        }
        var aq = ap.res;
        var ao = k(ap.mod, am);
        aa(J, function(at) {
            var au = at.name;
            if (au === ao) {
                ao = au + "/" + at.main;
                return false
            }
        });
        Z(am, M, function(at) {
            Z(ao, at, function(av, au) {
                ao = ao.replace(au, av)
            })
        });
        if (aq) {
            var an = u(ao);
            aq = an.normalize ? an.normalize(aq, function(at) {
                return af(at, am)
            }) : af(aq, am);
            ao += "!" + aq
        }
        return ao
    }
    function k(an, av) {
        if (an.indexOf(".") === 0) {
            var aw = av.split("/");
            var au = an.split("/");
            var am = aw.length - 1;
            var ao = au.length;
            var ar = 0;
            var ap = 0;
            pathLoop:
            for (var at = 0; at < ao; at++) {
                var aq = au[at];
                switch (aq) {
                case"..":
                    if (ar < am) {
                        ar++;
                        ap++
                    } else {
                        break pathLoop
                    }
                    break;
                case".":
                    ap++;
                    break;
                default:
                    break pathLoop
                }
            }
            aw.length = am - ar;
            au = au.slice(ap);
            return aw.concat(au).join("/")
        }
        return an
    }
    function W(an) {
        var am = an.split("!");
        if (am[0]) {
            return {
                mod: am[0],
                res: am[1]
            }
        }
        return null
    }
    function U(ar, ap, am) {
        var aq = [];
        for (var an in ar) {
            if (ar.hasOwnProperty(an)) {
                var ao = {
                    k: an,
                    v: ar[an]
                };
                aq.push(ao);
                if (ap) {
                    ao.reg = an === "*" && am ? /^/ : Y(an)
                }
            }
        }
        return aq
    }
    var Q;
    var w;
    function t() {
        if (Q) {
            return Q
        } else {
            if (w && w.readyState === "interactive") {
                return w
            } else {
                var am = document.getElementsByTagName("script");
                var ao = am.length;
                while (ao--) {
                    var an = am[ao];
                    if (an.readyState === "interactive") {
                        w = an;
                        return an
                    }
                }
            }
        }
    }
    var T = document.getElementsByTagName("head")[0];
    var d = document.getElementsByTagName("base")[0];
    if (d) {
        T = d.parentNode
    }
    function ak(am) {
        Q = am;
        d ? T.insertBefore(am, d) : T.appendChild(am);
        Q = null
    }
    function Y(am) {
        return new RegExp("^" + am + "(/|$)")
    }
    function aa(ap, ao) {
        if (ap instanceof Array) {
            for (var an = 0, am = ap.length; an < am; an++) {
                if (ao(ap[an], an) === false) {
                    break
                }
            }
        }
    }
    function m(an, am) {
        var ap = an.k || an.name;
        var ao = am.k || am.name;
        if (ao === "*") {
            return - 1
        }
        if (ap === "*") {
            return 1
        }
        return ao.length - ap.length
    }
    if (!b.define) {
        b.define = R;
        if (!b.require) {
            b.require = i
        }
        b.esl = i
    }
})(this);
define("jquery", [], function() {
    return window.jQuery
});
jQuery && jQuery.extend({
    stringify: function stringify(b) {
        if ("JSON" in window) {
            return JSON.stringify(b)
        }
        var l = typeof(b);
        if (l != "object" || b === null) {
            if (l == "string") {
                b = '"' + b + '"'
            }
            return String(b)
        } else {
            var d = {
                "\b": "\\b",
                "\t": "\\t",
                "\n": "\\n",
                "\f": "\\f",
                "\r": "\\r",
                '"': '\\"',
                "\\": "\\\\"
            };
            function i(n) {
                if (/["\\\x00-\x1f]/.test(n)) {
                    n = n.replace(/["\\\x00-\x1f]/g, function(o) {
                        var p = d[o];
                        if (p) {
                            return p
                        }
                        p = o.charCodeAt();
                        return "\\u00" + Math.floor(p / 16).toString(16) + (p%16).toString(16)
                    })
                }
                return '"' + n + '"'
            }
            function a(s) {
                var o = ["["], p = s.length, n, q, r;
                for (q = 0; q < p; q++) {
                    r = s[q];
                    switch (typeof r) {
                    case"undefined":
                    case"function":
                    case"unknown":
                        break;
                    default:
                        if (n) {
                            o.push(",")
                        }
                        o.push($.stringify(r));
                        n = 1
                    }
                }
                o.push("]");
                return o.join("")
            }
            switch (typeof b) {
            case"undefined":
                return "undefined";
            case"number":
                return isFinite(b) ? String(b) : "null";
            case"string":
                return i(b);
            case"boolean":
                return String(b);
            default:
                if (b === null) {
                    return "null"
                } else {
                    if (b instanceof Array) {
                        return a(b)
                    } else {
                        var m = ["{"], g = $.stringify, f, k;
                        for (var j in b) {
                            if (Object.prototype.hasOwnProperty.call(b, j)) {
                                k = b[j];
                                switch (typeof k) {
                                case"undefined":
                                case"unknown":
                                case"function":
                                    break;
                                default:
                                    if (f) {
                                        m.push(",")
                                    }
                                    f = 1;
                                    m.push(g(j) + ":" + g(k))
                                }
                            }
                        }
                        m.push("}");
                        return m.join("")
                    }
                }
            }
        }
    },
    format: function(d, a) {
        d = String(d);
        var b = Array.prototype.slice.call(arguments, 1), f = Object.prototype.toString;
        if (b.length) {
            b = b.length == 1 ? (a !== null && (/\[object Array\]|\[object Object\]/.test(f.call(a))) ? a : b) : b;
            return d.replace(/#\{(.+?)\}/g, function(g, j) {
                var i = b[j];
                if ("[object Function]" == f.call(i)) {
                    i = i(j)
                }
                return ("undefined" == typeof i ? "" : i)
            })
        }
        return d
    },
    subByte: function(m, b, g) {
        var f = [], k = m.split("");
        g = g || "…";
        for (var j = 0, d = k.length; j < d; j++) {
            if (k[j].charCodeAt(0) > 255) {
                f.push("*")
            }
            f.push(k[j])
        }
        if (b && b > 0 && f.length > b) {
            k = f.join("").substring(0, b - 1).replace(/\*/g, "") + g
        } else {
            return m
        }
        return k
    },
    getByteLength: function(j) {
        var d = [], g = j.split("");
        for (var f = 0, b = g.length; f < b; f++) {
            if (g[f].charCodeAt(0) > 255) {
                d.push("*")
            }
            d.push(g[f])
        }
        return d.length
    },
    _isValidKey: function(a) {
        return (new RegExp('^[^\\x00-\\x20\\x7f\\(\\)<>@,;:\\\\\\"\\[\\]\\?=\\{\\}\\/\\u0080-\\uffff]+\x24')).test(a)
    },
    setCookie: function(d, f, b) {
        f = encodeURIComponent(f);
        if (!jQuery._isValidKey(d)) {
            return 
        }
        b = b || {};
        var a = b.expires;
        if ("number" == typeof b.expires) {
            a = new Date();
            a.setTime(a.getTime() + b.expires)
        }
        document.cookie = d + "=" + f + (b.path ? "; path=" + b.path : "") + (a ? "; expires=" + a.toGMTString() : "") + (b.domain ? "; domain=" + b.domain : "") + (b.secure ? "; secure" : "")
    },
    getCookie: function(b) {
        var f = "";
        if (jQuery._isValidKey(b)) {
            var d = new RegExp("(^| )" + b + "=([^;]*)(;|\x24)"), a = d.exec(document.cookie);
            if (a) {
                f = a[2] || null;
                if ("string" == typeof f) {
                    f = decodeURIComponent(f);
                    return f
                }
            }
        }
        return null
    },
    removeCookie: function(b, a) {
        a = a || {};
        a.expires = new Date(0);
        jQuery.setCookie(b, "", a)
    }
});
function addEV(d, b, a) {
    if (window.attachEvent) {
        d.attachEvent("on" + b, a)
    } else {
        if (window.addEventListener) {
            d.addEventListener(b, a, false)
        }
    }
}
function _aMC(d) {
    var b = d, a =- 1;
    while (b = b.parentNode) {
        a = parseInt(b.getAttribute("id"));
        if (a > 0) {
            return a
        }
    }
}
function al_c(a) {
    while (a.tagName != "TABLE") {
        a = a.parentNode
    }
    return a.getAttribute("id")
}
function al_c2(b, a) {
    while (a--) {
        while ((b = b.parentNode).tagName != "TABLE") {}
    }
    return b.getAttribute("id")
}
function c(a) {
    var k = a.p1;
    if (a.fm == "alop"&&!("rsv_xpath" in a)) {
        if (k && G(k).getAttribute("srcid") == "6677") {} else {
            return true
        }
    }
    if (k&&!("p5" in a)) {
        a.p5 = k
    }
    var b = window.document.location.href, g = "", d = "", l = "", f = window["BD_PS_C" + (new Date()).getTime()] = new Image();
    for (v in a) {
        switch (v) {
        case"title":
            d = a[v].replace(/<[^<>]+>/g, "");
            if (d && d.length > 100) {
                d = d.substring(0, 100)
            }
            d = encodeURIComponent(d);
            break;
        case"mu":
        case"url":
            d = escape(a[v]);
            break;
        default:
            d = a[v]
        }
        g += "&" + v + "=" + d
    }
    if (!("mu" in a)) {
        try {
            if (("p2" in a) && G(a.p1).getAttribute("mu") && a.fm != "pl") {
                l = "&mu=" + escape(G(a.p1).getAttribute("mu"))
            }
        } catch (i) {}
    }
    var j = bds.comm.ubsurl + "?q=" + bds.comm.queryEnc + g + l + "&rsv_sid=" + bds.comm.sid + "&cid=" + bds.comm.cid + "&qid=" + bds.comm.queryId + "&t=" + new Date().getTime();
    if (bds.comm.inter) {
        j = j + "&rsv_inter=" + bds.comm.inter
    }
    if (bds && bds.comm && bds.comm.seinfo && bds.comm.seinfo.rsv_pstg) {
        j = j + "&rsv_pstg=" + bds.comm.seinfo.rsv_pstg
    }
    if (bds && bds.comm && bds.comm.resultPage) {
        j = j + "&rsv_iorr=1"
    } else {
        j = j + "&rsv_iorr=0"
    }
    if (bds && bds.comm && bds.comm.tn) {
        j = j + "&rsv_tn=" + bds.comm.tn
    }
    j += "&path=" + encodeURIComponent(b);
    f.src = j;
    return true
}
$(window).on("resize", function() {
    if ("pageState" in window && pageState != 0) {
        bds.util.setContainerWidth();
        bds.event.trigger("se.window_resize")
    }
});
(function() {
    var b = bds.util && bds.util.domain ? bds.util.domain.get("http://s1.bdstatic.com"): "http://s1.bdstatic.com";
    var a = bds.util && bds.util.domain ? bds.util.domain.get("http://ecma.bdimg.com"): "http://ecma.bdimg.com";
    require.config({
        baseUrl: b + "/r/www/cache/biz",
        packages: [{
            name: "ecma",
            location: a + "/public01"
        }
        ],
        paths: {
            aladdin: b + "/r/www/aladdin",
            ui: b + "/r/www/cache/amd/ui",
            "ui/config": b + "/r/www/cache/amd/ui/Control",
            "ui/lib": b + "/r/www/cache/amd/ui/Control",
            "ui/Control": b + "/r/www/cache/amd/ui/Control"
        },
        urlArgs: {
            "ui/ImgZoomHover": "20141104",
            "ui/ImgZoomHover1": "20141104",
            "ui/ImgZoomHover2": "20141104",
            "ui/ImgZoomHover3": "20141104"
        }
    })
})();
function TagQ(a, b) {
    return b.getElementsByTagName(a)
}
function h(b) {
    b.style.behavior = "url(#default#homepage)";
    b.setHomePage(bds.comm.domain);
    var a = window["BD_PS_C" + (new Date()).getTime()] = new Image();
    a.src = bds.comm.ubsurl + "?fm=hp&tn=" + bds.comm.tn + "&t=" + new Date().getTime()
}
function setHeadUrl(b) {
    var d = G("kw").value;
    d = encodeURIComponent(d);
    var a = b.href;
    a = a.replace(new RegExp("(" + b.getAttribute("wdfield") + "=)[^&]*"), "\x241" + d + "&ie=utf-8");
    b.href = a
}
bds.util.addStyle = function(b) {
    if (isIE) {
        var d = document.createStyleSheet();
        d.cssText = b
    } else {
        var a = document.createElement("style");
        a.type = "text/css";
        a.appendChild(document.createTextNode(b));
        document.getElementsByTagName("HEAD")[0].appendChild(a)
    }
};
bds.util.getContentRightHeight = function() {
    return ($("#content_right").get(0)) ? $("#content_right").get(0).offsetHeight : 0
};
bds.util.getContentLeftHeight = function() {
    return ($("#content_left").get(0)) ? $("#content_left").get(0).offsetHeight : 0
};
if (!window.A) {
    function G(a) {
        return document.getElementById(a)
    }
    window.bds = window.bds || {};
    bds.util = bds.util || {};
    bds.util.getWinWidth = function() {
        return window.document.documentElement.clientWidth
    };
    bds.util.setContainerWidth = function() {
        var g = G("container"), b = G("wrapper"), a = function(i, j) {
            j.className = j.className.replace(i, "")
        }, f = function(i, j) {
            j.className = (j.className + " " + i).replace(/^\s+/g, "")
        }, d = function(i, j) {
            return i.test(j.className)
        };
        if (bds.util.getWinWidth() < 1207) {
            if (g) {
                a(/\bcontainer_l\b/g, g);
                if (!d(/\bcontainer_s\b/, g)) {
                    f("container_s", g)
                }
            }
            if (b) {
                a(/\bwrapper_l\b/g, b);
                if (!d(/\bwrapper_s\b/, b)) {
                    f("wrapper_s", b)
                }
            }
            bds.comm.containerSize = "s"
        } else {
            if (g) {
                a(/\bcontainer_s\b/g, g);
                if (!d(/\bcontainer_l\b/, g)) {
                    f("container_l", g)
                }
            }
            if (b) {
                a(/\bwrapper_s\b/g, b);
                if (!d(/\bwrapper_l\b/, b)) {
                    f("wrapper_l", b)
                }
            }
            bds.comm.containerSize = "l"
        }
    };
    (function() {
        var d = [], i = false;
        var b = function(k, j) {
            try {
                k.call(j)
            } catch (l) {}
        }, f = function() {
            this.ids = [];
            this.has = true;
            this.list = [];
            this.logs = [];
            this.loadTimes = [];
            this.groupData = [];
            this.mergeFns = [];
            this._currentContainer = null
        };
        window.A = bds.aladdin = {};
        b(f, window.A);
        bds.ready = function(j) {
            if (typeof j != "function") {
                return 
            }
            if (i) {
                b(j)
            } else {
                d.push(j)
            }
        };
        bds.doReady = function() {
            i = true;
            while (d.length) {
                b(d.shift())
            }
        };
        bds.clearReady = function() {
            i = false;
            d = []
        };
        A.__reset = f;
        var a = (function() {
            var j = document.getElementsByTagName("script");
            return function() {
                var l = j[j.length - 1];
                if (window.currentScriptElem) {
                    l = window.currentScriptElem
                }
                var k = l;
                while (k) {
                    if (k.className) {
                        if (/(?:^|\s)result(?:-op)?(?:$|\s)/.test(k.className)) {
                            if (tplname = k.getAttribute("tpl")) {
                                return k
                            }
                        }
                    }
                    k = k.parentNode
                }
            }
        })(), g = function(j, m, l) {
            var n;
            if (!j.initIndex) {
                n = {
                    container: j,
                    data: {},
                    handlers: []
                };
                j.initIndex = A.groupData.length + 1;
                A.groupData.push(n)
            } else {
                n = A.groupData[j.initIndex - 1]
            }
            if (typeof m == "function") {
                n.handlers.push(m)
            } else {
                if (typeof m == "object") {
                    for (var o in m) {
                        if (m.hasOwnProperty(o)) {
                            n.data[o] = m[o]
                        }
                    }
                } else {
                    n.data[m] = l
                }
            }
        };
        A.init = A.setup = function(m, l) {
            if (m === undefined || m === null) {
                return 
            }
            var j = A._currentContainer || a();
            if (!j) {
                return 
            }
            g(j, m, l)
        };
        A.merge = function(k, j) {
            A.mergeFns.push({
                tplName: k,
                fn: j
            })
        }
    })()
}
function ns_c_pj(a, d) {
    var b = encodeURIComponent(window.document.location.href), k = "", i = "", l = "", g = bds.comm.queryEnc, f = bds && bds.util && bds.util.domain ? bds.util.domain.get("http://nsclick.baidu.com"): "http://nsclick.baidu.com", j = window["BD_PS_C" + (new Date()).getTime()] = new Image();
    for (v in a) {
        switch (v) {
        case"title":
            i = encodeURIComponent(a[v].replace(/<[^<>]+>/g, ""));
            break;
        case"url":
            i = encodeURIComponent(a[v]);
            break;
        default:
            i = a[v]
        }
        k += v + "=" + i + "&"
    }
    l = "&mu=" + b;
    j.src = f + "/v.gif?pid=201&" + (d || "") + k + "path=" + b + "&wd=" + g + "&rsv_sid=" + bds.comm.sid + "&t=" + new Date().getTime();
    return true
}
function ns_c(a) {
    return ns_c_pj(a, "pj=www&")
}
A.uiPrefix = "//www.baidu.com/cache/aladdin/ui/";
(function() {
    var b = window.bds.aladdin;
    var g = [];
    var l = {}, i = 0;
    var d = function(q, p) {
        try {
            q.call(p)
        } catch (r) {}
    };
    var f = function(p) {
        p.ajaxId=++i;
        l[p.ajaxId] = p
    };
    var n = function(p) {
        delete l[p.ajaxId]
    };
    var k = function(p) {
        if (!p.ajaxId) {
            return false
        }
        return l.hasOwnProperty(p.ajaxId)
    };
    var m = function(q) {
        var p = {};
        if (q) {
            try {
                var s = new Function("return " + q)();
                if (s) {
                    p = s
                }
            } catch (t) {}
        }
        return p
    }, a = function() {
        var r = $(".result-op").get().concat($(".result").get()), t = {};
        for (var q = 0, p, s; s = r[q]; q++) {
            if (p = s.getAttribute("tpl")) {
                if (t[p]) {
                    t[p].push(s)
                } else {
                    t[p] = [s]
                }
            }
        }
        return t
    };
    var o = [], j = [];
    b.addDisposeHandler = function(p) {
        j.push(p)
    };
    b.dispose = function() {
        while (o.length) {
            var r = o.shift();
            d(r.fn, r.obj)
        }
        var p = j;
        for (var q = 0; q < p.length; q++) {
            var r = p[q];
            d(r.fn, r.obj)
        }
    };
    b.__clearDispose = function() {
        o = [];
        j = []
    };
    b.addDisposeHandler({
        obj: l,
        fn: function() {
            for (var p in l) {
                if (l.hasOwnProperty(p)) {
                    delete l[p]
                }
            }
        }
    });
    b._Aladdin = function() {
        this.p1 = 0;
        this.mu = null
    };
    $.extend(b._Aladdin.prototype, {
        _init: function() {
            var r = this, p;
            p = r.container;
            var q = m(r.container.getAttribute("data-click"));
            r.p1 = q.p1 || p.id;
            r.mu = q.mu || p.getAttribute("mu");
            r.srcid = q.rsv_srcid || p.getAttribute("srcid")
        },
        q: function(q, p) {
            p = p || "";
            return $(this.container).find(p + "." + q).get()
        },
        qq: function(q, p) {
            return this.q(q, p)[0]
        },
        find: function(p) {
            return window.jQuery(p, this.container)
        },
        ready: function() {
            $(document).ready.apply(this, arguments)
        },
        ajax: function(C, F, u) {
            var D = b.AJAX;
            var H =+ new Date();
            var s = u.params || {};
            var w = {
                query: C,
                co: u.co || "",
                resource_id: F,
                t: H
            };
            $.extend(w, D.PARAMS);
            $.extend(w, s);
            var C = $.param(w);
            var r = D.API_URL + "?" + C;
            var B = function() {
                var p = new Image();
                p.src = $.format(D.ERR_URL, {
                    url: r
                });
                b.logs.push(p)
            };
            var z = new Date().getTime();
            var y = function(p) {
                var t = new Date().getTime() - z;
                var I = {
                    fm: "opendataajax",
                    srcid: F,
                    time: t,
                    status: p
                };
                ns_c(I)
            };
            var E = function(p) {
                if (!k(E)) {
                    return 
                }
                q();
                if (p.status == 0) {
                    u.success(p.data)
                } else {
                    u.error && u.error(p.status);
                    B()
                }
                y(0)
            };
            var x = function() {
                if (!k(x)) {
                    return 
                }
                q();
                u.timeout && u.timeout();
                B();
                y(1)
            };
            var q = function() {
                n(E);
                n(x)
            };
            f(E);
            f(x);
            $.ajax({
                url: r,
                scriptCharset: D.PARAMS.oe,
                timeout: D.TIMEOUT,
                dataType: "jsonp",
                jsonp: "cb",
                success: E,
                error: x
            })
        }
    });
    b.AJAX = {
        API_URL: bds.util.domain && bds.util.domain.get ? bds.util.domain.get("http://opendata.baidu.com/api.php"): "http://opendata.baidu.com/api.php",
        ERR_URL: bds.util.domain && bds.util.domain.get ? bds.util.domain.get("http://open.baidu.com/stat/al_e.gif?ajax_err_url=#{url}"): "http://open.baidu.com/stat/al_e.gif?ajax_err_url=#{url}",
        PARAMS: {
            ie: "utf8",
            oe: "gbk",
            cb: "op_aladdin_callback",
            format: "json",
            tn: "baidu"
        },
        TIMEOUT: 6000
    };
    g.push(function(p) {
        var q = /msie (\d+\.\d+)/i.test(navigator.userAgent) ? (document.documentMode||+RegExp["\x241"]): undefined;
        if (q) {
            var r = document.charset;
            $.each(p.container.getElementsByTagName("form"), function(t, u) {
                var w = function() {
                    var x = u.acceptCharset;
                    if (x && x.toString().toUpperCase() != "UNKNOWN" && x != document.charset) {
                        document.charset = x;
                        setTimeout(function() {
                            document.charset = r
                        }, 1000)
                    }
                };
                $(u).on("submit", w);
                var s = u.submit;
                u.submit = function() {
                    w();
                    try {
                        s.call(u)
                    } catch (x) {
                        s()
                    }
                }
            })
        }
    });
    b.__runAla = function() {
        var p = a();
        $.each(b.mergeFns, function(s, q) {
            var r = p[q.tplName];
            if (r) {
                $.each(r, function(t, u) {
                    b._currentContainer = u;
                    q.fn();
                    b._currentContainer = null
                })
            }
        });
        $.each(b.groupData, function(t, s) {
            var w = new b._Aladdin(), r, u, x;
            w.container = s.container;
            w.data = s.data;
            w._init();
            b.list.push(w);
            var q = s.handlers;
            r = new Date();
            while (q.length) {
                d(q.shift(), w)
            }
            if (typeof w.dispose == "function") {
                o.push({
                    obj: w,
                    fn: w.dispose
                });
                w.dispose = null
            }
            u = new Date(), x = {
                srcid: w.srcid
            };
            x.tpl = w.container.getAttribute("tpl");
            x.time = u - r;
            b.loadTimes.push(x);
            $.each(g, function(y, z) {
                z.call(w, w)
            })
        })
    }
})();
(function() {
    var g = window.A, b = {}, j = {}, s = {}, p = document, n = p.getElementsByTagName("head")[0], i = false, f = ["baidu"], q = false, d = g.baidu, m = function() {};
    var l = {
        "*": function(t, u) {
            return u + "?v=2014010100"
        },
        scrollbarv: function(t, u) {
            return u + "?v=20140121"
        },
        likeshare4: function(t, u) {
            return u + "?v=20140116"
        },
        mboxsingleton: function(t, u) {
            return u + "?v=20141008"
        },
        sms: function(t, u) {
            return u + "?v=20140507"
        },
        tab: function(t, u) {
            return u + "?v=20140117"
        },
        tabs: function(t, u) {
            return u + "?v=20140116"
        },
        musicplayer: function(t, u) {
            return u + "?v=20140916"
        },
        slider: function(t, u) {
            return u + "?v=20140123"
        },
        suggestion: function(t, u) {
            return u + "?v=20140924"
        },
        tabs5: function(t, u) {
            return u + "?v=20140911"
        },
        tabx: function(t, u) {
            return u + "?v=20140117"
        },
        dropdown1: function(t, u) {
            return u + "?v=20140117"
        },
        dropdown21: function(t, u) {
            return u + "?v=20140227"
        },
        advert: function(t, u) {
            return u + "?v=20140523"
        },
        honourCard: function(t, u) {
            return u + "?v=20140926"
        },
        share: function(t, u) {
            return u + "?v=20141107"
        },
        qHotCity: function(t, u) {
            return u + "?v=20140806"
        },
        mapapi: function(t, u) {
            return u + "?v=20140829"
        },
        qunarfilters: function(t, u) {
            return u + "?v=20141114"
        },
        renderIframe3: function(t, u) {
            return u + "?v=20140917"
        },
        ALD_feedback: function(t, u) {
            return u + "?v=20140901"
        }
    };
    $(document).ready(function() {
        i = true
    });
    g.addDisposeHandler({
        obj: g,
        fn: function() {
            for (var t in s) {
                if (s.hasOwnProperty(t)) {
                    var u = s[t];
                    while (u.length) {
                        u.pop()
                    }
                }
            }
        }
    });
    function a(u, z) {
        var x = typeof u === "string" ? u.split(/\s*,\s*/): u;
        if (x.length > 1) {
            if (z) {
                a(x.shift(), function() {
                    if (x.length > 0) {
                        a(x, z)
                    }
                })
            } else {
                while (x.length) {
                    a(x.shift())
                }
            }
            return 
        }
        u = x[0];
        if (u === "jquery" && window.jQuery) {
            !g.ui.jquery && (g.ui.jquery = window.jQuery);
            z && z();
            return 
        }
        var y = u.replace(/\./g, "/");
        var t = u.replace(/^[\s\S]*\./, "");
        var w = g.uiPrefix + y + "/" + t;
        if (y.search("style/") == 0) {
            o(w + ".css", z)
        } else {
            w += ".js";
            if (l.hasOwnProperty(u)) {
                if (typeof l[u] == "function") {
                    w = l[u](u, w)
                } else {
                    if (typeof l[u] == "string") {
                        w = l[u]
                    }
                }
            } else {
                if (l.hasOwnProperty("*")) {
                    w = l["*"](u, w)
                }
            }
            if (z) {
                r(w, z)
            } else {
                k(w)
            }
        }
    }
    a.cache = b;
    function o(u, w) {
        w = w || m;
        if (u in b) {
            w();
            return 
        }
        var t = p.createElement("link");
        t.rel = "stylesheet";
        t.type = "text/css";
        t.href = u;
        t.setAttribute("data-for", "A.ui");
        n.appendChild(t);
        b[u] = 1;
        w()
    }
    function k(t) {
        if (i) {
            r(t, m);
            return 
        }
        if (t in b) {
            return 
        }
        p.write('<script charset="gb2312" type="text/javascript" src="' + t + '"><\/script>');
        b[t] = 1
    }
    function r(w, x) {
        x = x || m;
        if (w in b) {
            x();
            return 
        }
        if (w in j) {
            s[w].push(x);
            return 
        }
        j[w] = 1;
        var u = s[w] = [x];
        var t = p.createElement("script");
        t.type = "text/javascript";
        t.charset = "gb2312";
        t.onload = t.onreadystatechange = function() {
            if ((!this.readyState || this.readyState === "loaded" || this.readyState === "complete")) {
                while (u.length) {
                    u.shift()()
                }
                delete j[w];
                b[w] = 1;
                t.onload = t.onreadystatechange = null
            }
        };
        t.src = w;
        t.setAttribute("data-for", "A.ui");
        n.insertBefore(t, n.firstChild)
    }
    g.uicss = function(t) {
        o(g.uiPrefix + t)
    };
    g.uijs = function(t, u) {
        r(g.uiPrefix + t, u)
    };
    g.uijsPathMap = function(t) {
        $.extend(l, t)
    };
    g.use = a;
    g.ui = g.ui || {};
    g.addCssText = function(x) {
        var B = "opui-style-tag-id", u = "data-for", t = "A.ui", w = document.getElementById(B);
        if (!w) {
            w = document.createElement("style");
            w.setAttribute("type", "text/css");
            w.setAttribute(u, t);
            w.id = B;
            document.getElementsByTagName("head")[0].appendChild(w)
        }
        try {
            var y = document.createTextNode(x);
            w.appendChild(y)
        } catch (z) {
            if (w.styleSheet) {
                w.styleSheet.cssText += x
            }
        }
    };
    $(window).on("swap_end", function() {
        var u = /MSIE\s?6/.test(window.navigator.userAgent);
        var t = function(w, y, x) {
            $(w).each(function(C, B) {
                var D = $(B), z = new Image(), E = D.attr("src");
                z.onload = function() {
                    var K = y, H = x, J = this.width, F = this.height, I = (J / F) / (K / H);
                    if (I > 1) {
                        if (J > K) {
                            J = K
                        } else {
                            J = "auto"
                        }
                        F = "auto"
                    } else {
                        if (F > H) {
                            F = H
                        } else {
                            F = "auto"
                        }
                        J = "auto"
                    }
                    D.css({
                        height: F,
                        width: J
                    });
                    z.onload = null;
                    z = null
                };
                z.src = E
            })
        };
        if (u) {
            t("img.result-left-img", 98, 121)
        }
        $(".c-feedback").bind("click", function() {
            var w = this;
            g.use("ALD_feedback", function() {
                var D = "right", x, z, C = $(w);
                if (C.parents("#content_left").length) {
                    D = "left";
                    z = C.parents(".result-op"), x = z.attr("srcid")
                } else {
                    if (C.parents("#content_right").length) {
                        z = C.parents("#con-ar")
                    }
                }
                var y = {
                    query: bds.comm.query,
                    srcid: x,
                    target: z,
                    username: bds.comm.username,
                    flag: D
                };
                var B = g.ui.ALD_feedback(y);
                g.addDisposeHandler({
                    obj: B,
                    fn: B.dispose
                })
            })
        })
    })
})();
$(window).on("swap_begin", function() {
    A.dispose();
    A.__reset();
    A.__clearDispose()
});
$(window).on("swap_dom_ready", function() {
    bds.ready(A.__runAla);
    bds.doReady()
});
bds.event = new function() {
    var i = {};
    function d(j, l) {
        if (typeof l == "function" || l instanceof Function) {
            var k = a(j);
            i[k.name] = i[k.name] || [];
            i[k.name].push({
                prod: k.prod,
                callback: l
            })
        }
    }
    function g(l, o) {
        var n = a(l), k = i[n.name] || [], j = 0;
        while (j < k.length) {
            var m = k[j];
            if (o === m.callback && f(n.prod, m.prod)) {
                k.splice(j, 1)
            } else {
                j++
            }
        }
    }
    function b(j, m) {
        var q = a(j), k = i[q.name] || [], r = {
            data: m,
            eventId: j
        };
        for (var n = 0, o = k.length; n < o; n++) {
            var l = k[n];
            try {
                if (f(l.prod, q.prod)) {
                    l.callback.call(this, r)
                }
            } catch (p) {}
        }
    }
    function f(j, k) {
        return new RegExp("^" + j.replace(/\./g, "\\.").replace(/\*/g, ".+") + "$").test(k)
    }
    function a(j) {
        var k = j.match(/(.+)\.(.+)/);
        if (k && k[2]) {
            return {
                prod: k[1],
                name: k[2]
            }
        } else {
            return {}
        }
    }
    this.on = d;
    this.off = g;
    this.trigger = b;
    this.events = i
};
(function(A) {
    var baidu = window.baidu;
    var LOG_CLASS = ["TITLE", "LINK", "IMG", "BTN", "INPUT", "OTHERS"];
    var C_LOG_CLASS = ["btn"];
    var contentLeft, contentRight, contentTop;
    function clickDebug(e) {}
    window.initResultClickLog = function() {
        contentLeft = $("#content_left").get(0);
        contentRight = $("#con-ar").get(0);
        contentTop = $("#con-at").get(0);
        if (A.has) {
            var aladdin_tables = $(".result-op").get(), srcid;
            $.each(aladdin_tables, function(i, v) {
                if (srcid = v.getAttribute("srcid")) {
                    A.ids.push([v.id, srcid])
                }
            })
        }
        bindP5()
    };
    $(document).ready(function() {
        bindLogEvent()
    });
    function bindP5() {
        var item, index = (bds.comm.pageNum - 1) * bds.comm.pageSize + 1, leftItems = (contentLeft && contentLeft.children) || [], rightItems = (contentRight && contentRight.children) || [], topItems = (contentTop && contentTop.children) || [], isResult = function(o) {
            return (o.nodeType == 1 && o.className && /\bresult(\-op)?\b/.test(o.className))
        }, isFrame = function(o) {
            return (o.nodeType == 1 && o.className && /\bc\-frame\b/.test(o.className))
        }, setClickData = function(wrap, data) {
            var sData = wrap.getAttribute("data-click") || "{}";
            try {
                var oData = eval("(" + sData + ")");
                sData = $.stringify($.extend(oData, data));
                wrap.setAttribute("data-click", sData)
            } catch (e) {
                clickDebug(e)
            }
        }, bindP5ClickData = function(items) {
            for (var i = 0, l = items.length; i < l; i++) {
                item = items[i];
                if (isResult(item)) {
                    setClickData(item, {
                        p5: index++
                    })
                } else {
                    if (isFrame(item)) {
                        try {
                            var frameItems = item.children[0].children;
                            for (var j = 0, lj = frameItems.length; j < lj; j++) {
                                var frameItem = frameItems[j];
                                if (isResult(frameItem)) {
                                    setClickData(frameItem, {
                                        p5: index++
                                    })
                                }
                            }
                        } catch (e) {
                            clickDebug(e)
                        }
                    }
                }
            }
            index = (bds.comm.pageNum - 1) * bds.comm.pageSize + 1
        };
        bindP5ClickData(leftItems);
        bindP5ClickData(rightItems);
        bindP5ClickData(topItems)
    }
    function getXPath(node, wrap, path) {
        path = path || [];
        wrap = wrap || document;
        if (node === wrap) {
            return path
        }
        if (node.parentNode !== wrap) {
            path = getXPath(node.parentNode, wrap, path)
        }
        if (node.previousSibling) {
            var count = 1;
            var sibling = node.previousSibling;
            do {
                if (sibling.nodeType == 1 && sibling.nodeName == node.nodeName) {
                    count++
                }
                sibling = sibling.previousSibling
            }
            while (sibling)
            }
        if (node.nodeType == 1) {
            path.push(node.nodeName.toLowerCase() + (count > 1 ? count : ""))
        }
        return path
    }
    function bindLogEvent() {
        $body = $("body");
        $body.on("mousedown", function(e) {
            var e = window.event || e, t = e.srcElement || e.target, $t = $(t);
            try {
                var $parent = $t, fm, wrap;
                while ($parent.length&&!($parent.hasClass("result") || $parent.hasClass("result-op") || $parent.hasClass("xpath-log"))) {
                    $parent = $parent.parent()
                }
                if (!$parent.length) {
                    return 
                }
                if ($parent.hasClass("result-op")) {
                    fm = "alop"
                } else {
                    if ($parent.hasClass("result")) {
                        fm = "as"
                    }
                }
                wrap = $parent.get(0);
                var xpath = getXPath(t, wrap);
                if (check(xpath, t, wrap)) {
                    log(xpath, t, wrap, fm)
                }
            } catch (e) {
                clickDebug(e)
            }
        })
    }
    function getType(xpath, t, wrap) {
        var node = t, cs = LOG_CLASS, cl = cs.length, clc = C_LOG_CLASS, clcl = clc.length, xstr = xpath.join(" "), i = 0;
        while (node !== wrap) {
            for (i = 0; i < cl; i++) {
                if ($(node).hasClass("OP_LOG_" + cs[i])) {
                    return cs[i].toLowerCase()
                }
            }
            for (i = 0; i < clcl; i++) {
                if ($(node).hasClass("c-" + clc[i])) {
                    return clc[i]
                }
            }
            node = node.parentNode
        }
        if (/\bh3\d*\b/.test(xstr)) {
            return "title"
        }
        if (/\ba\d*\b/.test(xstr)) {
            if (/\bimg\d*\b/.test(xstr)) {
                return "img"
            }
            return "link"
        }
        if (/\b(input|select|button|textarea|datalist)\d*\b/.test(xstr)) {
            return "input"
        }
        if (/\blabel\d*\b/.test(xstr) && t.getElementsByTagName("input").length > 0) {
            return "input"
        }
        return ""
    }
    function check(xpath, t, wrap) {
        if (A.LOGTOOL) {
            A.LOGTOOL.call(t, xpath, t, wrap);
            return false
        }
        return true
    }
    function log(xpath, t, wrap, fm) {
        if (t.getAttribute("data-nolog") != null) {
            return 
        }
        var type = getType(xpath, t, wrap);
        if (!type) {
            return false
        }
        if (type == "title"&&!/\ba\d*\b/.test(xpath)) {
            return false
        }
        var nourl = "http://nourl.ubs.baidu.com";
        var mu = wrap.getAttribute("mu") || nourl;
        if (mu == nourl) {
            var h3 = wrap.getElementsByTagName("h3");
            if (h3 && h3[0]) {
                var a = h3[0].getElementsByTagName("a");
                mu = (a && a[0]) ? a[0].href : mu
            }
        }
        var l = xpath.length, url, p = t, srcid = wrap.getAttribute("srcid");
        var title = "";
        var tag = t.nodeType == 1 ? t.tagName.toLowerCase(): "";
        if (type == "input") {
            if (/input|textarea/.test(tag)) {
                title = t.value;
                if (t.type && t.type.toLowerCase() == "password") {
                    title = ""
                }
            } else {
                if (/select|datalist/.test(tag)) {
                    if (t.children.length > 0) {
                        var index = t.selectedIndex || 0;
                        title = t.children[index>-1 ? index: 0].innerHTML
                    }
                } else {
                    title = t.innerHTML || t.value || ""
                }
            }
        } else {
            if (tag == "img") {
                title = t.title
            }
            if (!title) {
                while (l > 0) {
                    l--;
                    if (/^a\d*\b/.test(xpath[l])) {
                        url = p.href;
                        title = p.innerHTML;
                        if (p.getAttribute("data-nolog") != null) {
                            return 
                        }
                        break
                    } else {
                        if (p.className && (/OP_LOG_/.test(p.className))) {
                            title = p.innerHTML;
                            break
                        }
                        p = p.parentNode
                    }
                }
            }
        }
        title = $.trim(title);
        if (!url || url.slice( - 1) === "#" ||!(/^http/.test(url))) {
            url = mu
        }
        var data = {
            rsv_xpath: xpath.join("-") + "(" + type + ")",
            title: title,
            url: url,
            rsv_height: wrap.offsetHeight,
            rsv_width: wrap.offsetWidth,
            rsv_tpl: wrap.getAttribute("tpl")
        };
        var rewritedatakey = {
            url: 1,
            title: 1
        };
        if (wrap.id && wrap.id.match(/^\d+$/)) {
            data.p1 = wrap.id
        }
        if (srcid) {
            data.rsv_srcid = srcid
        }
        var ext_data, attr, is_fm_null;
        p = t;
        do {
            if (p.getAttribute("data-nolog") != null) {
                return 
            }
            if (ext_data = p.getAttribute("data-click")) {
                try {
                    ext_data = (new Function("return " + ext_data))();
                    for (attr in ext_data) {
                        if (attr == "fm" && ext_data.fm === null) {
                            is_fm_null = true
                        }
                        if (ext_data.hasOwnProperty(attr) && ((typeof data[attr] == "undefined") || rewritedatakey[attr])) {
                            data[attr] = ext_data[attr]
                        }
                    }
                } catch (e) {
                    clickDebug(e)
                }
            }
            p = p.parentNode
        }
        while (p && p !== wrap.parentNode);
        for (var i in data) {
            if (data[i] === null) {
                delete data[i]
            }
        }
        if (type == "title") {
            if ("mu" in data) {
                delete data.mu
            }
        } else {
            if (!data.mu) {
                data.mu = mu
            }
        }
        if (is_fm_null) {
            if ("fm" in data) {
                delete data.fm
            }
        } else {
            if (type == "input") {
                data.fm = "beha";
                data.url = nourl
            }
            if (!data.fm) {
                data.fm = fm
            }
            if (!data.fm) {
                return 
            }
        }
        window.c(data)
    }
})(window.bds.aladdin);
for (ai in al_arr) {
    al_arr[ai]()
}
$(document).ready(function() {
    var a;
    $(document).on("click", "h3.t>a", function(g) {
        g = window.event || g;
        var d = $("#wrapper_wrapper"), b = $(this).closest(".c-container"), f = b.length ? b.find(".c-recommend").eq(0): [];
        if (!g.ctrlKey && f.length && f.css("display") === "none") {
            a = setTimeout(function() {
                d.find(".c-recommend").hide();
                f.show()
            }, 150)
        }
    });
    $(window).on("swap_begin", function() {
        this.clearTimeout(a)
    })
});
window.onunload = function() {};
function addEV(d, b, a) {
    if (window.attachEvent) {
        d.attachEvent("on" + b, a)
    } else {
        if (window.addEventListener) {
            d.addEventListener(b, a, false)
        }
    }
}
bds.se.openime = function(a) {
    if (!window.bdime) {
        $.ajax({
            cache: true,
            dataType: "script",
            url: "http://s1.bdstatic.com/r/www/cache/static/plugins/ime_a41ccd23.js",
            success: function() {
                if (a) {
                    openIme.set("py", true)
                }
            }
        })
    } else {
        openIme.set("py", true)
    }
};
if (/\bbdime=[12]/.test(document.cookie)) {
    bds.se.openime(false)
} else {
    $(window).one("swap_end", function() {
        var b = function() {
            var f = "";
            if (bds.comm.upn && bds.comm.upn.os === "windows") {
                if (bds.comm.upn.browser !== "baiduclient") {
                    f = $("<span class=\"shouji\"><a href=\"http://w.x.baidu.com/go/mini/132/10000011\" onmousedown=\"return ns_c({'fm':'behs','tab':'bdclient'})\">添加百度到桌面，搜索更便捷！</a></span>")
                }
                if (f) {
                    if (bds.comm.containerSize === "s") {
                        f.hide()
                    }
                    f.insertAfter("#mHolder");
                    $(window).on("container_resize", function(i, g) {
                        var j = $("#mHolder").nextAll(".shouji").eq(0);
                        if (j.length) {
                            if (j.css("display") === "none" && g === "l") {
                                j.show()
                            }
                            if (g === "s") {
                                j.hide()
                            }
                        }
                    })
                }
            }
        };
        if (bds.comm.username) {
            var d = bds.util && bds.util.domain ? bds.util.domain.get("http://api.open.baidu.com"): "http://api.open.baidu.com", a = d + "/pae/common/api/invitation?type=get";
            $.ajax({
                url: a,
                dataType: "jsonp",
                jsonp: "cb",
                success: function(f) {
                    if (f && f.data && f.data.status === "1") {
                        require.config({
                            paths: {
                                bear: "//www.baidu.com/aladdin/js/bear/bear.min"
                            }
                        });
                        require(["bear"], function(g) {
                            var i = new g();
                            i.init(bds.comm.userid || bds.comm.username || bds.comm.user || $.getCookie("BAIDUID"), bds.comm.query)
                        })
                    } else {
                        b()
                    }
                },
                error: function() {
                    b()
                },
                timeout: 3000
            })
        } else {
            b()
        }
    })
}
if (bds && bds.comm&&!bds.comm.containerSize) {
    bds.comm.containerSize = "s"
}
bds.util.setContainerWidth = function() {
    var f = $("#container"), a = $("#wrapper"), b = bds.util.getWinWidth();
    var d = bds.comm.containerSize;
    if (b < 1217) {
        f.removeClass("container_l container_xl").addClass("container_s");
        a.removeClass("wrapper_l").addClass("wrapper_s");
        bds.comm.containerSize = "s"
    } else {
        if (b >= 1217) {
            f.removeClass("container_s container_xl").addClass("container_l");
            a.removeClass("wrapper_s").addClass("wrapper_l");
            bds.comm.containerSize = "l"
        } else {
            return 
        }
    }
    if (d != bds.comm.containerSize) {
        $(window).trigger("container_resize", bds.comm.containerSize)
    }
};
var bds = bds || {};
bds.se = bds.se || {};
bds.se.tip = bds.se.tip || {};
bds.comm.tipZIndex = 220;
bds.comm.tips = [];
bds.se.tip = function(a) {
    this.init = function() {
        this.op = {
            target: a.target || {},
            mode: a.mode || "over",
            title: a.title || null,
            content: a.content || null,
            uncontrolled: (a.uncontrolled) ? true: false,
            arrow: {
                has: 1,
                offset: 10
            },
            close: a.close || 0,
            align: a.align || "left",
            offset: {
                x: 10,
                y: 20
            },
            arrowSize: 16
        };
        if (a.arrow) {
            this.op.arrow.has = (a.arrow.has == 0) ? 0 : 1;
            this.op.arrow.offset = (a.arrow.offset >= 0) ? a.arrow.offset : 10
        }
        if (a.offset) {
            this.op.offset.x = (a.offset.x || a.offset.x == 0) ? a.offset.x : 10;
            this.op.offset.y = (a.offset.y || a.offset.y == 0) ? a.offset.y : 20
        }
        this.ext = a.ext || {};
        this.dom = $("<div>", {
            "class": "c-tip-con"
        });
        this.visible = false;
        this.rendered = false;
        this.isAuto = (this.op.align === "auto") ? true : false;
        this.bindEvent()
    };
    this.render = function() {
        if (this.op.close) {
            this.enableCloseIcon()
        }
        if (this.op.title) {
            this.setTitle(this.op.title)
        }
        if (this.op.content) {
            this.setContent(this.op.content)
        }
        if (this.op.arrow.has) {
            this.enableArrow()
        }
        $("#c-tips-container").append(this.dom)
    };
    this.bindEvent = function() {
        this.delay = {
            overIcon: null,
            outIcon: null,
            overDom: null,
            outDom: null
        };
        if (this.op.mode == "over") {
            var b = this;
            $(b.op.target).on("mouseenter", function() {
                window.clearTimeout(b.delay.outIcon);
                window.clearTimeout(b.delay.outDom);
                b.delay.overIcon = setTimeout(function() {
                    b.show()
                }, 200)
            });
            b.dom.on("mouseenter", function() {
                window.clearTimeout(b.delay.outIcon);
                window.clearTimeout(b.delay.outDom);
                b.delay.overDom = setTimeout(function() {
                    b.show()
                }, 200)
            });
            $(b.op.target).on("mouseleave", function() {
                window.clearTimeout(b.delay.overIcon);
                window.clearTimeout(b.delay.overDom);
                b.delay.outIcon = setTimeout(function() {
                    b.hide()
                }, 200)
            });
            b.dom.on("mouseleave", function() {
                window.clearTimeout(b.delay.overIcon);
                window.clearTimeout(b.delay.overDom);
                b.delay.outIcon = setTimeout(function() {
                    b.hide()
                }, 200)
            })
        } else {
            if (this.op.mode == "none") {
                var b = this;
                b.show()
            }
        }
    };
    this.enableArrow = function() {
        var b = $("<div>", {
            "class": "c-tip-arrow"
        }).html("<em></em><ins></ins>").appendTo(this.dom);
        this.arrow = b
    };
    this.enableCloseIcon = function() {
        var d = this;
        var b = $("<div>", {
            "class": "c-tip-close"
        }).html("<i class='c-icon c-icon-close'></i>").appendTo(this.dom).click(function() {
            d.hide()
        });
        this.close = b
    };
    this.setTitle = function(b) {
        if (b.nodeType) {
            var d = $("<h3>", {
                "class": "c-tip-title"
            }).append(b).appendTo(this.dom)
        } else {
            var d = $("<h3>", {
                "class": "c-tip-title"
            }).html(b).appendTo(this.dom)
        }
        this.title = d
    };
    this.setContent = function(d) {
        var b = $("<div>").html(d).appendTo(this.dom);
        this.content = b
    };
    this.setArrow = function(b) {
        if (b) {
            if (b.offset >= 0) {
                this.op.arrow.offset = b.offset
            }
        }
        if (this.op.arrow.has && this.arrow) {
            switch (this.op.align) {
            case"left":
                this.arrow.css({
                    left: this.op.arrow.offset + "px"
                });
                break;
            case"right":
                this.arrow.css({
                    right: this.op.arrow.offset + 16 + "px"
                });
                break;
            default:
                break
            }
        }
    };
    this.setOffset = function(b) {
        if (b) {
            this.op.offset.x = (b.x || b.x == 0) ? b.x : this.op.offset.x;
            this.op.offset.y = (b.y || b.y == 0) ? b.y : this.op.offset.y
        }
        switch (this.op.align) {
        case"left":
            var d = $(this.getTarget()).offset();
            this.getDom().css({
                top: d.top + this.op.offset.y + "px",
                left: d.left - this.op.offset.x + "px"
            });
            break;
        case"right":
            var d = $(this.getTarget()).offset();
            this.getDom().css({
                top: d.top + this.op.offset.y + "px",
                left: d.left + this.op.offset.x + $(this.getTarget()).width() - this.getDom().width() + "px"
            });
            break;
        default:
            break
        }
    };
    this.autoOffset = function(o) {
        var d = {
            w: this.dom.outerWidth(),
            h: this.dom.outerHeight()
        }, m = $(this.getTarget()), n = m.offset(), i = {
            w: m.outerWidth(),
            h: m.outerHeight()
        }, l = $(window), f = l.scrollTop(), k = {
            w: l.width(),
            h: l.height()
        }, b = {
            left: 0,
            top: 0
        }, j = {}, g;
        if ((k.h + f - i.h - n.top) > d.h) {
            b.top = i.h + n.top + this.op.arrowSize / 2;
            this.arrow.removeClass("c-tip-arrow-down")
        } else {
            if (n.top - f > d.h) {
                b.top = n.top - d.h - this.op.arrowSize / 2;
                this.arrow.addClass("c-tip-arrow-down")
            } else {
                b.top = i.h + n.top + this.op.arrowSize / 2;
                this.arrow.removeClass("c-tip-arrow-down")
            }
        }
        g = n.left + i.w / 2 - this.op.arrow.offset - this.op.arrowSize / 2;
        b.left = g;
        if (b.left > 0 && (b.left + d.w) > k.w) {
            b.left = n.left + i.w / 2 - d.w + this.op.arrow.offset + this.op.arrowSize / 2;
            j.right = this.op.arrow.offset + this.op.arrowSize;
            j.left = "auto";
            if (b.left < 0) {
                b.left = g;
                j.left = this.op.arrow.offset;
                j.right = "auto"
            }
        } else {
            j.left = this.op.arrow.offset;
            j.right = "auto"
        }
        this.dom.css(b);
        this.arrow.css(j)
    };
    this.enable = function() {};
    this.disable = function() {};
    this.destroy = function() {};
    this.show = function() {
        if (!this.visible) {
            this.onShow();
            if (!this.rendered) {
                bds.comm.tips.push(this);
                this.render();
                this.rendered = true
            }
            if (this.isAuto) {
                this.autoOffset()
            } else {
                this.setOffset();
                this.setArrow()
            }
            this.dom.css({
                "z-index": bds.comm.tipZIndex
            });
            bds.comm.tipZIndex++;
            this.dom.css({
                display: "block"
            });
            this.visible = true
        }
    };
    this.hide = function() {
        if (this.visible) {
            this.dom.css({
                display: "none"
            });
            this.onHide();
            this.visible = false
        }
    };
    this.onShow = a.onShow || function() {};
    this.onHide = a.onHide || function() {};
    this.getTarget = function() {
        return this.op.target
    };
    this.getDom = function() {
        return this.dom
    };
    this.init()
};
bds.event.trigger("se.api_tip_ready");
$(document).mousedown(function(b) {
    b = b || window.event;
    var a = b.target || b.srcElement;
    while (a && a.tagName && a != document.body && a.tagName.toLowerCase() != "html") {
        if (a.className == "c-tip-con") {
            break
        }
        a = a.parentNode
    }
    if (a && a.className != "c-tip-con") {
        $(bds.comm.tips).each(function() {
            if (!this.op.uncontrolled) {
                if (this.op.close) {
                    this.hide()
                }
            }
        })
    }
});
var sethfPos = sethfPos || 0;
(function() {
    var q = "//www.baidu.com/", n = navigator.userAgent.indexOf("MSIE")!=-1&&!window.opera, r = Math.random() * 100, w = "百度一下，你就知道", d = "";
    window.fa = function(z) {
        try {
            if (window.sidebar) {
                window.sidebar.addPanel(w, q, "")
            } else {
                if (window.opera && window.print) {
                    z.setAttribute("rel", "sidebar");
                    z.setAttribute("href", q);
                    z.setAttribute("title", w);
                    z.click()
                } else {
                    window.external.AddFavorite(q, w)
                }
            }
        } catch (y) {}
    };
    function f(z) {
        if (z) {
            var y = z.parentNode;
            if (y) {
                y.style.marginBottom = "20px";
                y.style.marginTop = "2px"
            }
        }
    }
    if (n) {
        try {
            var x = /se /gi.test(navigator.userAgent);
            var o = /AppleWebKit/gi.test(navigator.userAgent) && /theworld/gi.test(navigator.userAgent);
            var l = /theworld/gi.test(navigator.userAgent);
            var p = /360se/gi.test(navigator.userAgent);
            var a = /360chrome/gi.test(navigator.userAgent);
            var g = /greenbrowser/gi.test(navigator.userAgent);
            var t = /qqbrowser/gi.test(navigator.userAgent);
            var m = /tencenttraveler/gi.test(navigator.userAgent);
            var k = /maxthon/gi.test(navigator.userAgent);
            var u = /krbrowser/gi.test(navigator.userAgent);
            var b = false;
            try {
                b =+ external.twGetVersion(external.twGetSecurityID(window)).replace(/\./g, "") > 1013
            } catch (s) {}
            if (x || b || o || l || p || a || g || t || m || k || u) {
                var j = sethfPos ? document.getElementById("set_f"): document.getElementById("setf");
                if (j) {
                    j.style.display = "inline";
                    if (sethfPos) {
                        f(j);
                        d = "favorites"
                    }
                }
            } else {
                var i = sethfPos ? document.getElementById("set_h"): document.getElementById("seth");
                if (i) {
                    i.style.display = "inline";
                    if (sethfPos) {
                        f(i);
                        d = "home"
                    }
                }
            }
        } catch (s) {}
    } else {
        var j = sethfPos ? document.getElementById("set_f"): document.getElementById("setf");
        if (j) {
            j.style.display = "inline"
        }
        if (sethfPos) {
            f(j);
            d = "favorites"
        }
    }
    if (d && sethfPos) {
        ns_c({
            fm: "sethf_show",
            tab: d
        })
    }
})();
function user_c(i) {
    var g = "", f = "", a = "", b = "", k = encodeURIComponent(window.document.location.href), d = window["BD_PS_C" + (new Date()).getTime()] = new Image(), j = bds && bds.util && bds.util.domain ? bds.util.domain.get("http://nsclick.baidu.com"): "http://nsclick.baidu.com";
    for (v in i) {
        switch (v) {
        case"title":
            a = encodeURIComponent(i[v].replace(/<[^<>]+>/g, ""));
            break;
        case"url":
            a = encodeURIComponent(i[v]);
            break;
        default:
            a = i[v]
        }
        g += v + "=" + a + "&"
    }
    b = "&mu=" + k;
    d.src = j + "/v.gif?pid=201&pj=psuser&" + g + "path=" + k + "&wd=" + f + "&t=" + new Date().getTime();
    return true
}
function initPassV3() {
    bds.se.passv3 = passport.pop.init({
        apiOpt: {
            loginType: 1,
            product: "mn",
            u: window.document.location.href,
            safeFlag: 0,
            staticPage: location.protocol + "//www.baidu.com/cache/user/html/v3Jump.html"
        },
        cache: false,
        tangram: true,
        authsite: ["qzone", "tsina"],
        authsiteCfg: {
            act: "implicit",
            display: "popup",
            jumpUrl: location.protocol + "//www.baidu.com/cache/static/user/html/xd.html",
            onBindSuccess: function(b, d) {
                var a = decodeURIComponent(d.passport_uname || d.displayname);
                bds.se.login.success(a);
                return false
            }
        },
        onLoginSuccess: function(b) {
            b.returnValue = false;
            var a = b.rsp.data.userName || b.rsp.data.mail || b.rsp.data.phoneNumber;
            bds.se.login.success(a)
        },
        onSubmitStart: function(a) {},
        onSubmitedErr: function(a) {},
        onSystemErr: function(a) {},
        onShow: function() {
            var b = $(".tang-pass-login .pass-reglink"), a = b.attr("href");
            b.attr("href", a.replace(/\&subpro=[0-9a-zA-Z%]*([\&]*)/g, ""))
        },
        onHide: function() {},
        onDestroy: function() {}
    })
}
if (!bds.comm.user) {
    $.getScript(location.protocol + "//passport.baidu.com/passApi/js/uni_login_wrapper.js?cdnversion=" + new Date().getTime(), initPassV3)
}
bds.se.loginCallbackFunc = null;
bds.se.login = {
    init: function() {
        this.setUserInfo();
        var a = this;
        bds.comm.loginAction.push(function(b, d) {
            a.setUserInfo(d)
        })
    },
    setUserInfo: function(b) {
        var a = b || bds.comm.user;
        if (!a) {
            return 
        }
        $("#lb").replaceWith('<a href="http://i.baidu.com" class="username">' + escapeHTML(bds.comm.username) + '<i class="c-icon"></i></a>')
    },
    open: function(b, a) {
        bds.se.loginCallbackFunc = b || function() {
            window.document.location.reload(true)
        };
        bds.se.passv3.show()
    },
    success: function(a) {
        if (!bds.comm) {
            return 
        }
        bds.comm.user = a;
        bds.comm.username = a;
        window.bdUser = a;
        bds.se.passv3.hide();
        bds.se.loginCallbackFunc.call(window, 1, a);
        for (var b = 0; b < bds.comm.loginAction.length; b++) {
            bds.comm.loginAction[b].call(window, 1, a)
        }
    }
};
bds.se.login.init();
window._invoke_login = function(b, a) {
    bds.se.login.open(b, a)
};
function isp_hijack(g) {
    var i = document.getElementById("wrapper"), b, a = false, d, f;
    if (!bds.comm.query) {
        a = true
    }
    if (g.stat == 1) {
        b = document.createElement("div");
        b.innerHTML = '<div style="zoom:1;_margin-left:1024px;"><div style="position:relative;_float:left;_margin-left:-1024px;"><div style="width:100%;min-width:1024px;"><div style="border:2px solid #fd9162;zoom:1;overflow:hidden;padding:0 0 6px 12px;"><div style="position:relative;width:100%;*overflow:auto;padding-top:10px;"><div style="height:18px;margin-bottom:6px;"><i class="c-icon" style="width:18px;height:18px;background-position:-168px -72px;"></i><strong style="display:inline-block;margin-left:8px;font-size:14px;color:#666;">百度提示您：</strong></div><span style="display:block;color:#333;text-indent:26px;font-size:13px;">我们发现当前您可能受到异常广告弹窗的影响，通常这是受第三方恶意劫持导致。使用 <a href="http://shadu.baidu.com/landingpage/competing.html?from=10064" target="_blank" style="color:#0000D0;text-decoration:underline">防恶意广告专版杀毒软件</a>，可有效改善您的上网体验，免受恶意广告的困扰。</span><a id="isp-close-btn" style="display:inline-block;width:9px;height:9px;position:absolute;top:6px;right:6px;background:url(../global/img/wsCloseBtn2.png) no-repeat;" href="javascript:void(0);"></a></div></div></div></div></div>';
        if (!a) {
            i.style.position = "relative";
            document.getElementById("u").style.top = 0;
            b.style.margin = "-6px 0 8px 0";
            document.body.insertBefore(b, i)
        } else {
            i.insertBefore(b, i.children[0])
        }
        d = document.getElementById("isp-close-btn");
        f = d.parentNode.getElementsByTagName("a")[0];
        d.onclick = function() {
            if (a) {
                i.removeChild(b)
            } else {
                document.body.removeChild(b);
                i.style.position = "";
                document.getElementById("u").style.top = "4px"
            }
        };
        d.onmousedown = function() {
            ns_c({
                fm: "behs",
                tab: "tj_notice",
                cont: "jcbro",
                action: "close",
                area: "topbar"
            })
        };
        f.onmousedown = function() {
            ns_c({
                fm: "behs",
                tab: "tj_notice",
                cont: "jcbro",
                action: "click",
                area: "topbar"
            })
        };
        ns_c({
            fm: "behs",
            tab: "tj_notice",
            cont: "jcbro",
            action: "show",
            area: "topbar"
        })
    }
}(function() {
    function a() {
        var d, f = "http://isphijack.baidu.com/index.php?cb=isp_hijack", j = [];
        if (top.location != self.location) {
            try {
                var b = top.document.getElementsByTagName("frame");
                var l = top.document.getElementsByTagName("iframe");
                for (var g = 0; g < b.length; g++) {
                    j.push(b[g].getAttribute("src"))
                }
                for (var g = 0; g < l.length; g++) {
                    j.push(l[g].getAttribute("src"))
                }
            } catch (k) {}
            ns_c({
                fm: "frm",
                top: encodeURIComponent(top.location.href),
                furls: encodeURIComponent(j.join("|"))
            });
            if (j) {
                d = document.createElement("script");
                d.src = f + "&urls=" + encodeURIComponent(j.join("|")) + "&t=" + ( + new Date());
                document.body.appendChild(d)
            }
        }
    }
    $(a)
})();
try {
    if (window.console && window.console.log) {
        console.log("一张网页，要经历怎样的过程，才能抵达用户面前？\n一位新人，要经历怎样的成长，才能站在技术之巅？\n探寻这里的秘密；\n体验这里的挑战；\n成为这里的主人；\n加入百度，加入网页搜索，你，可以影响世界。\n");
        console.log("请将简历发送至 %c ps_recruiter@baidu.com（ 邮件标题请以“姓名-应聘XX职位-来自console”命名）", "color:red");
        console.log("职位介绍：http://dwz.cn/hr2013")
    }
} catch (e) {}
var bds = bds || {};
bds.se = bds.se || {};
bds.se.tool = bds.se.tool || {};
bds.comm.host = {
    bfe: "//www.baidu.com/tools",
    favo: bds.util.domain && bds.util.domain.get ? bds.util.domain.get("http://i.baidu.com"): "http://i.baidu.com",
    share: bds.util.domain && bds.util.domain.get ? bds.util.domain.get("http://bdimg.share.baidu.com/static/api/js/custom/resultshare.js"): "http://bdimg.share.baidu.com/static/api/js/custom/resultshare.js",
    report: "http://jubao.baidu.com",
    koubei: "http://koubei.baidu.com"
};
bds.se.tool = function(item) {
    this.init = function() {
        this.render()
    };
    this.render = function() {
        var ops = eval("(" + item.getAttribute("data-tools") + ")");
        var toolsDom = $("<div>", {
            "class": "c-tip-menu"
        });
        var toolsList = $("<ul>");
        var toolsFavo = $("<li>");
        var toolsFavoLink = $("<a>").html("收藏");
        toolsFavoLink.on("mousedown", function() {
            bds.se.tool.favo(ops, item.getAttribute("id"));
            ns_c({
                fm: "tools",
                tab: "favo"
            })
        });
        toolsFavoLink.on("mouseover", function() {
            $(this).css("background-color", "#ebebeb")
        });
        toolsFavoLink.on("mouseout", function() {
            $(this).css("background-color", "#fff")
        });
        toolsFavo.append(toolsFavoLink);
        toolsList.append(toolsFavo);
        var toolsShare = $("<li>");
        var toolsShareLink = $("<a>").html("分享");
        toolsShareLink.on("mousedown", function() {
            bds.se.tool.share(ops, item);
            ns_c({
                fm: "tools",
                tab: "share"
            })
        });
        toolsShareLink.on("mouseover", function() {
            $(this).css("background-color", "#ebebeb")
        });
        toolsShareLink.on("mouseout", function() {
            $(this).css("background-color", "#fff")
        });
        toolsShare.append(toolsShareLink);
        toolsList.append(toolsShare);
        var toolsKoubei = $("<li>").html("<a target=\"_blank\" onmousedown=\"ns_c({'fm': 'tools','tab':'koubei'})\" href=\"" + bds.comm.host.bfe + "?url=" + encodeURIComponent(ops.url) + "&jump=" + encodeURIComponent(bds.comm.host.koubei + "/womc/p/sentry?title=" + encodeURIComponent(ops.title) + "&q=" + encodeURIComponent(bds.comm.query)) + '&key=surl">评价</a>');
        toolsList.append(toolsKoubei);
        var toolsReport = $("<li>").html("<a target=\"_blank\" onmousedown=\"ns_c({'fm': 'tools','tab':'report'})\" href=\"" + bds.comm.host.bfe + "?url=" + encodeURIComponent(ops.url) + "&jump=" + encodeURIComponent(bds.comm.host.report + "/jubao/accu/?title=" + encodeURIComponent(ops.title) + "&q=" + encodeURIComponent(bds.comm.query)) + '&key=surl">举报</a>');
        toolsList.append(toolsReport);
        toolsDom.append(toolsList);
        var tTip = new bds.se.tip({
            target: $(".c-icon", item)[0],
            mode: "none",
            align: "left",
            offset: {
                x: 33
            },
            arrow: {
                has: 1,
                offset: 30
            },
            content: toolsDom,
            ext: {
                category: "tools"
            }
        });
        tTip.onShow = function() {
            ns_c({
                fm: "tools",
                tab: "show"
            })
        }
    };
    this.init()
};
bds.se.tool.share = function(b, a) {
    this.op = b || {};
    this.init = (function(f, d) {
        $.getScript(bds.comm.host.share, function() {
            $(bds.comm.tips).each(function() {
                if (!this.op.uncontrolled) {
                    this.hide()
                }
            });
            var g = new bds.se.tip({
                target: $(".c-icon", d)[0],
                mode: "none",
                offset: {
                    x: 33
                },
                arrow: {
                    has: 0
                },
                close: 1,
                content: '<div class="c-tools-share" style="width:200px;"></div>'
            });
            var i = $(".c-tools-share", g.dom.get(0))[0];
            __bdshare.render({
                boxEle: i,
                url: f.url,
                txt: f.title + " -- 分享自百度搜索"
            })
        })
    })(this.op, a)
};
bds.se.tool.favo = function(d, b) {
    this.op = d || {};
    this.init = function(k, j) {
        if (k) {
            var f = document.createElement("script");
            var g = bds.comm.host.bfe, i = bds.comm.host.favo;
            f.src = g + "?url=" + encodeURIComponent(k.url) + "&jump=" + encodeURIComponent(i + "/myfavorite/set?irt=1&t=" + encodeURIComponent(k.title) + "&id=" + encodeURIComponent(j) + "&c=bds.se.tool.favo.succ") + "&key=url";
            document.body.appendChild(f)
        }
    };
    if (bds.comm.user) {
        this.init(this.op, b)
    } else {
        if (bds.se.login && bds.se.login.open) {
            var a = this;
            bds.se.login.open(function(g, f) {
                if (g == 1) {
                    a.init(a.op, b)
                }
            })
        }
    }
};
bds.se.tool.favo.succ = function(json) {
    if (json.suc) {
        if (json.status) {
            switch (json.status) {
            case 302:
                if (bds.se.login && bds.se.login.open) {
                    bds.se.login.open(function(stat, user) {
                        if (stat == 1) {
                            bds.se.tool.favo(eval("(" + $("#" + json.id)[0].getAttribute("data-tools") + ")"), json.id)
                        }
                    })
                }
                break;
            case 5:
                var succContent = '<div class="c-tip-notice">';
                succContent += '<h3 class="c-tip-notice-fail">收藏失败，请稍后再试</h3>';
                succContent += "</div>";
                break
            }
        }
    } else {
        if (json.status) {
            var succContent = '<div class="c-tip-notice">';
            succContent += '<h3 class="c-tip-notice-succ">已收藏至：</h3>';
            succContent += "<ul>";
            switch (json.status) {
            case 2:
                succContent += '<li class="c-tip-item-succ"><i class="c-icon c-icon-success"></i>个人中心“<a href="http://i.baidu.com/my/collect" target="_blank">我的收藏</a>”</li>';
                succContent += '<li class="c-tip-item-succ"><i class="c-icon c-icon-success"></i>百度首页“<a href="http://www.baidu.com" target="_blank">我的导航</a>”</li>';
                break;
            case 3:
                succContent += '<li class="c-tip-item-succ"><i class="c-icon c-icon-success"></i>个人中心“<a href="http://i.baidu.com/my/collect" target="_blank">我的收藏</a>”</li>';
                succContent += '<li class="c-tip-item-fail"><i class="c-icon c-icon-fail"></i>百度首页“<a href="http://www.baidu.com" target="_blank">我的导航</a>”</li>';
                break;
            case 4:
                succContent += '<li class="c-tip-item-fail"><i class="c-icon c-icon-fail"></i>个人中心“<a href="http://i.baidu.com/my/collect" target="_blank">我的收藏</a>”</li>';
                succContent += '<li class="c-tip-item-succ"><i class="c-icon c-icon-success"></i>百度首页“<a href="http://www.baidu.com" target="_blank">我的导航</a>”</li>';
                break;
            default:
                break
            }
            succContent += "</ul>";
            succContent += "</div>"
        }
    }
    $(bds.comm.tips).each(function() {
        if (!this.op.uncontrolled) {
            this.hide()
        }
    });
    new bds.se.tip({
        target: $(".c-icon", document.getElementById(json.id))[0],
        offset: {
            x: 33
        },
        arrow: {
            has: 0
        },
        mode: "none",
        arrow: {
            has: 0
        },
        close: 1,
        content: succContent
    })
};
var bds = bds || {};
bds.se = bds.se || {};
bds.se.tools = bds.se.tools || {};
bds.se.tools = function() {
    var a = delayHideOnIcon = delayShowOnTip = delayHideOnTip = {};
    $("#container").delegate(".c-tools", "mouseover", function() {
        var b = this;
        window.clearTimeout(delayHideOnIcon);
        window.clearTimeout(delayHideOnTip);
        a = setTimeout(function() {
            var d = 1;
            $(bds.comm.tips).each(function(f) {
                if (this.getTarget() == $(".c-icon", b)[0]) {
                    d = 0;
                    this.show();
                    return false
                }
            });
            if (d) {
                tools = new bds.se.tool(b)
            }
        }, 200)
    }).delegate(".c-tools", "mouseout", function() {
        window.clearTimeout(a);
        window.clearTimeout(delayShowOnTip);
        var b = this;
        delayHideOnIcon = setTimeout(function() {
            $(bds.comm.tips).each(function(d) {
                if (this.getTarget() == $(".c-icon", b)[0]) {
                    this.hide();
                    return false
                }
            })
        }, 200)
    });
    $("#c-tips-container").delegate(".c-tip-con", "mouseover", function() {
        var b = this;
        window.clearTimeout(delayHideOnIcon);
        window.clearTimeout(delayHideOnTip);
        delayShowOnTip = setTimeout(function() {
            $(bds.comm.tips).each(function(d) {
                if (this.getDom().get(0) == b && this.ext.category && this.ext.category == "tools") {
                    this.show();
                    return false
                }
            })
        }, 200)
    }).delegate(".c-tip-con", "mouseout", function() {
        window.clearTimeout(a);
        window.clearTimeout(delayShowOnTip);
        var b = this;
        delayHideOnTip = setTimeout(function() {
            $(bds.comm.tips).each(function(d) {
                if (this.getDom().get(0) == b && this.ext.category && this.ext.category == "tools") {
                    this.hide();
                    return false
                }
            })
        }, 200)
    })
};
bds.se.tools();
var bds = bds || {};
bds.se = bds.se || {};
bds.se.slide = function(b) {
    var i = this, d = true, g = true, a = {}, f;
    this._default = {
        target: $("#lg"),
        src: "",
        frames: 59,
        width: 270,
        height: 129,
        duration: 100,
        offsetLeft: 0,
        isAutoPlay: false,
        isPad: false,
        delay: 0,
        repeats: 1,
        action: ""
    };
    this.timer = [];
    this.op = $.extend({}, i._default, b);
    this.init = function() {
        i.createDom();
        if (bds.comm.ishome && i.op.target.length) {
            i.bindEvent()
        }
    };
    this.createDom = function() {
        var k = '<div style="position:absolute;top:0;left:50%;background:url(#{0}) no-repeat #{1};cursor:#{2};width:#{3}px;height:#{4}px;margin-left:-#{5}px;display:none;"></div>', l = (i.op.action.indexOf("click")!==-1) ? "pointer": "default", n = i.op.offsetLeft + "px 0", m = i.op.target.find("map"), j = m.length ? m.find("area").eq(0): "", l = j ? "pointer": l;
        k = $.format(k, i.op.src, n, l, i.op.width, i.op.height, i.op.width / 2);
        if (i.op.target.css("position") === "static") {
            i.op.target.css({
                position: "relative",
                width: "100%"
            })
        }
        i.op.target.append(k);
        f = i.op.target.find("div").eq(0);
        if (i.op.isPad) {
            f.css("background-size", (i.op.width * i.op.frames / 2) + "px " + i.op.height + "px")
        }
        if (j.length) {
            f.wrap('<a href="' + j.attr("href") + '" target="' + j.attr("target") + '"></a>');
            if (j.attr("title")) {
                f.attr("title", j.attr("title"))
            }
        }
        a = {
            "background-position": n,
            cursor: l
        }
    };
    this.bindEvent = function() {
        var l = i.op.action.split("|");
        for (var o = 0, k = l.length; o < k; o++) {
            if (l[o] === "hover") {
                f.hover(function(q) {
                    var p = $(this), r;
                    if (d) {
                        if (q.type == "mouseenter") {
                            r = f.css("background-position");
                            r = parseInt(r) - i.op.width;
                            f.css("background-position", r + "px 0")
                        } else {
                            if (q.type == "mouseleave") {
                                f.css("background-position", a["background-position"])
                            }
                        }
                    }
                })
            }
            if (l[o] === "click") {
                f.on("click", function() {
                    var p = $(this);
                    if (g) {
                        i.play()
                    }
                    return false
                })
            }
        }
        var n = $("#lg area");
        if (n.length && n.attr("onmousedown")) {
            f.on("mousedown", function() {
                return (new Function(n.attr("onmousedown")))()
            })
        }
        if (i.op.isAutoPlay) {
            var m = new Image();
            m.src = i.op.src;
            if (!m.complete) {
                m.onload = j
            } else {
                j()
            }
        }
        function j() {
            f.show();
            i.play()
        }
    };
    this.play = function() {
        i.disableAction();
        i.resetToClick();
        i.animation()
    };
    this.disableAction = function() {
        g = false;
        d = false
    };
    this.enableAction = function() {
        g = true;
        d = true
    };
    this.resetToClick = function() {
        f.css({
            "background-position": i.op.offsetLeft + "px 0",
            cursor: "default"
        })
    };
    this.reset = function() {
        f.css(a)
    };
    this.animation = function() {
        var k = 0, j = 0, l, m = f;
        (function() {
            l = arguments.callee;
            i.timer.push(setTimeout(function() {
                var n = m.css("background-position").split(" "), n = n[0] ? parseInt(n[0])*-1: 0;
                m.css("background-position", - (n + i.op.width) + "px 0");
                j++;
                if (j >= i.op.frames) {
                    i.dispose();
                    k++;
                    if (i.op.repeats !== 0 && k >= i.op.repeats) {
                        i.reset();
                        i.enableAction();
                        j = null;
                        k = null
                    } else {
                        j = 0;
                        i.reset();
                        i.timer.push(setTimeout(l, i.op.duration))
                    }
                } else {
                    i.timer.push(setTimeout(arguments.callee, i.op.duration))
                }
            }, i.op.delay))
        })()
    };
    this.dispose = function(k) {
        k = k || i.timer;
        for (var l = 0, j = k.length; l < j; l++) {
            window.clearTimeout(k[l])
        }
        k.length = 0
    };
    this.clear = function() {
        i.op.repeats =- 1;
        i.dispose()
    };
    this.init()
};
var bds = bds || {};
bds.se = bds.se || {};
bds.se.banner = function(a, d, b) {
    this.init = function() {
        b = b || {};
        this.$dom_panel = $(a);
        this.hintText = d;
        this.hintIcon = b.iconClass || "";
        this.downUrl = b.downUrl || "";
        this.hintCookie = b.cookie || "";
        this.showNum = (this.hintCookie && $.getCookie(this.hintCookie)) ? Number($.getCookie(this.hintCookie)) : 0;
        this.nscTab = b.nscTab || "";
        this.ishome = (bds.comm && bds.comm.ishome == 1) ? 1 : 0;
        if (a && d && this.showNum < 5&&!$(".baiduapp_banner")[0]&&!$(".res_top_banner")[0]) {
            this.show()
        }
    };
    this.show = function() {
        this.render();
        this.showNum += 1;
        $.setCookie(this.hintCookie, this.showNum, {
            expires: 30 * 24 * 60 * 60 * 1000
        });
        this.$dom_panel.prepend(this.bannerHtml);
        if (this.ishome != 1) {
            this.headFloat()
        }
        this.bindEvent();
        ns_c({
            fm: "behs",
            tab: ((this.ishome == 1) ? "tj_" : "") + "baidu_" + (this.nscTab ? this.nscTab : "topbanner") + "show"
        })
    };
    this.render = function() {
        var f = [];
        f = f.concat(['<div class="res_top_banner">', '<i class="c-icon ' + (this.hintIcon ? this.hintIcon : "res_top_banner_icon") + '"></i>', "<span>" + this.hintText + "</span>", (this.downUrl) ? '<a href="' + this.downUrl + '" class="res_top_banner_download">立即体验</a>': "", '<i class="c-icon res_top_banner_close"></i>', "</div>"]);
        this.bannerHtml = f.join("")
    };
    this.headFloat = function() {
        var g = $("#head"), j = $(window), f = $(".res_top_banner");
        var i = g.css("position");
        $(window).scroll(function() {
            var l = f.height() || 0, k = j.scrollTop();
            if (k <= l) {
                g.attr("style", "position:absolute;")
            } else {
                g.attr("style", "top:0px;_top:" + l + "px;")
            }
        })
    };
    this.bindEvent = function() {
        var f = $(".res_top_banner"), g = this;
        $(".res_top_banner_download", f).on("mousedown", function() {
            g.hintCookie && $.setCookie(g.hintCookie, 5, {
                expires: 30 * 24 * 60 * 60 * 1000
            });
            ns_c({
                fm: "behs",
                tab: ((g.ishome == 1) ? "tj_" : "") + "baidu_" + (g.nscTab ? g.nscTab : "topbanner") + "down"
            })
        });
        $(".res_top_banner_close", f).on("mousedown", function() {
            f.detach();
            g.hintCookie && $.setCookie(g.hintCookie, 5, {
                expires: 30 * 24 * 60 * 60 * 1000
            });
            ns_c({
                fm: "behs",
                tab: ((g.ishome == 1) ? "tj_" : "") + "baidu_" + (g.nscTab ? g.nscTab : "topbanner") + "close"
            })
        });
        $(window).on("swap_begin", function() {
            f.detach()
        })
    };
    this.init()
};
(function() {
    $(window).on("swap_end", function() {
        var a = ["union", "union2baidu", "union_cpro", "union_nosearch", "redbull", "hao123"], f = ["union", "union2baidu", "union_cpro", "union_nosearch", "redbull"], d = bds.comm.upn;
        if (bds.comm.tng && $.inArray(bds.comm.tng, a)==-1) {
            if (d && d.browser && d.browser == "msie" && d.ie && (d.ie == "6" || d.ie == "7" || d.ie == "8")) {
                var b = (d.ie == "6") ? "您的浏览器采用的IE6内核已停止维护，推荐升级到更快更安全的百度浏览器！": "您的IE浏览器版本较低，即将停止更新维护，建议升级到更快、更安全的百度浏览器。";
                if (bds.comm.samBDClientFlag) {
                    bds.se.banner($("body")[0], "添加百度到桌面，感受最便捷的搜索！", {
                        iconClass: "res_top_banner_icon_baiduapp",
                        downUrl: "http://w.x.baidu.com/go/mini/132/10000029",
                        cookie: "H_PS_BCBANNER",
                        nscTab: "bdclient"
                    })
                } else {
                    bds.se.banner($("body")[0], b, {
                        downUrl: "http://w.x.baidu.com/go/mini/8/21010002",
                        cookie: "H_PS_BBANNER",
                        nscTab: "browser"
                    })
                }
            }
        } else {
            if (bds.comm.tng && $.inArray(bds.comm.tng, f)!=-1 && bds.comm.tnuka != 1) {
                if (d && d.browser && d.os == "windows" && d.browser != "baiduclient") {
                    bds.se.banner($("body")[0], "添加百度到桌面，感受最便捷的搜索！", {
                        iconClass: "res_top_banner_icon_baiduapp",
                        downUrl: "http://w.x.baidu.com/go/mini/132/10000008",
                        cookie: "H_PS_BABANNER",
                        nscTab: "baiduapp"
                    })
                }
            }
        }
    })
})();
bds.se.safeTip = (function() {
    var selfCSS = [".unsafe_content{margin-top:4px;}", "a.unsafe_ico_new{color:#b95b07;}", ".safe_icons{width:60px;line-height:15px;font-zise:12px;color:#666;text-align:center;display:inline-block;vertical-align:top}", ".safe_icons_bd{width:16px;height:16px;display:inline-block;background:url(//www.baidu.com/cache/spam/img/safe-icons-1.1.png) no-repeat -17px 0;position:relative;z-index:0}", ".safe_icons_qq{width:16px;height:16px;display:inline-block;background:url(//www.baidu.com/cache/spam/img/safe-icons-1.1.png) no-repeat -173px 0;position:relative;z-index:0}", ".safe_icons_hs{width:16px;height:16px;display:inline-block;background:url(//www.baidu.com/cache/spam/img/safe-icons-1.1.png) no-repeat -68px 0;position:relative;z-index:0}", ".safe_icons_js{width:16px;height:16px;display:inline-block;background:url(//www.baidu.com/cache/spam/img/safe-icons-1.1.png) no-repeat 0 0;position:relative;z-index:0}", ".safe_icons_sc{width:16px;height:16px;display:inline-block;background:url(//www.baidu.com/cache/spam/img/safe-icons-1.1.png) no-repeat -51px 0;position:relative;z-index:0}", ".safe_icons_fail{width:14px;height:14px;display:inline-block;background:url(//www.baidu.com/cache/spam/img/safe-icons-1.1.png)  no-repeat -159px 0;position:absolute;left:9px;top:7px}"];
    function init() {
        bds.util.addStyle(selfCSS.join(""));
        var num_unsafe = 0, tj_which = [], data_tpl = "", data_id = [];
        var unsafe = $(".unsafe_ico_new").get();
        for (var i = 0; i < unsafe.length; i++) {
            var obj = unsafe[i];
            var s_data = eval("(" + obj.getAttribute("data-safe") + ")");
            data_id.push(obj.getAttribute("data-id"));
            data_tpl = obj.getAttribute("data-tpl");
            var s_type = {
                qq: "0",
                bd: "0",
                sc: "0",
                js: "0",
                hs: "0"
            };
            if (s_data) {
                s_item = s_data.hintItem;
                for (var j = 0; j < s_item.length; j++) {
                    s_type[s_item[j]] = "1"
                }
                for (var key in s_type) {
                    if (s_type[key] == "1") {
                        tj_which.push(key)
                    }
                }
            }
            while (obj.className.indexOf("result")==-1) {
                obj = obj.parentNode
            }
            if (obj.className.indexOf("result")!=-1) {
                var h3 = obj.getElementsByTagName("h3");
                if (h3) {
                    var links = h3[0].getElementsByTagName("a");
                    if (links) {
                        var html = links[0].innerHTML;
                        links[0].href = unsafe[i].href;
                        links[0].innerHTML = html
                    }
                }
            }
            num_unsafe++;
            var safedata = eval("(" + unsafe[i].getAttribute("data-safe") + ")");
            var tip_id = unsafe[i].getAttribute("data-id");
            var tip_tpl = unsafe[i].getAttribute("data-tpl");
            setTipCon(unsafe[i], safedata, tip_id, tip_tpl)
        }
        if (num_unsafe > 0) {
            ns_c({
                tab: "safetip",
                num_unsafe: num_unsafe,
                prd: tj_which.join("|"),
                hintId: data_id,
                hintTpl: data_tpl
            })
        }
    }
    function setTipCon(obj, data, tip_id, tip_tpl) {
        var safeName = {
            qq: "电脑<br>管家",
            bd: "百度安<br>全检测",
            sc: "SCANV<br>安全中心",
            js: "金山<br>云安全",
            hs: "小红伞"
        };
        var groupName = {
            qq: 0,
            js: 0,
            bd: 0,
            sc: 0,
            hs: 0
        }, nameHtml = [], typeHtml = [], html = "";
        if (data == "") {
            return 
        }
        var group = data.hintItem;
        for (var i = 0; i < group.length; i++) {
            groupName[group[i]] = 1
        }
        for (var key in groupName) {
            if (groupName[key] == 1) {
                nameHtml.push('<span class="safe_icons"><span class="safe_icons_' + key + '"><span class="safe_icons_fail"></span></span><br>' + safeName[key] + "</span>")
            }
        }
        html = '<div class="c-tip-info"><strong><em>' + data.hintLabelMid + "</em>&nbsp;&nbsp;" + data.hintLabelSuf + "</strong>";
        html += '<p class="c-gap-top-small c-gap-bottom-small">' + nameHtml.join("") + "</p></div>";
        new bds.se.tip({
            target: obj,
            title: data.hintLabelPre + "：",
            content: html,
            offset: {
                x: - 20,
                y: 25
            },
            onShow: function() {
                ns_c({
                    tab: "safetip",
                    safe: 0,
                    which: data.hintItem.join("|"),
                    hintId: tip_id,
                    hintTpl: tip_tpl
                })
            }
        })
    }
    return {
        init: init
    }
})();
var bds = bds || {};
bds.se = bds.se || {};
bds.se.trust = bds.se.trust || {};
bds.se.trust = function() {
    var o = 4;
    var p = [];
    var n = [];
    if (bds.util && bds.util.domain && bds.util.domain.get) {
        var d = bds.util.domain.get("tag.baidu.com")
    } else {
        var d = "http://tag.baidu.com"
    }
    var j = null;
    var m = null;
    function q() {
        p = [], n = [];
        $(".c-trust").each(function() {
            var s = $(this);
            var r = this.getAttribute("data_key");
            if (s.parent(".c-icons-inner").length == 0) {
                s.wrap("<span class='c-icons-outer'><span class='c-icons-inner'></span></span>")
            }
            if ($.inArray(r, p)==-1) {
                p.push(this.getAttribute("data_key"))
            }
            if ($.inArray(this, n)==-1) {
                n.push(this)
            }
        });
        $(".c-trust-as").each(function() {
            m = $.parseJSON($(this).attr("hint-data"));
            if (m) {
                j = $(this);
                l(m, $(this).attr("hint-type"))
            }
        });
        if (p.length < 1) {
            return 
        }
        k()
    }
    function k() {
        $.getJSON(d + "/?urls=" + p.join(",") + "&sid=" + bds.comm.sid + "&qid=" + bds.comm.qid + "&v=" + o + "&callback=?", b)
    }
    function b(r) {
        if (r.code != 0) {
            return 
        }
        $(n).each(function() {
            var s = this.getAttribute("data_key");
            m = r.data[s];
            if (!m) {
                return 
            }
            j = $(this);
            j.html("");
            if (m.vstar && m.vstar.hint && m.vstar.hint.length > 0) {
                f(m.vstar.hint[0].vlevel, m.vstar.url)
            }
            if (m.medical) {
                g()
            }
            if (m.aviation) {
                i()
            }
        })
    }
    function f(u, r) {
        var s = $("<span>", {
            "class": "c-vline"
        });
        var t = $("<a>", {
            "class": "c-icon c-icon-v" + u,
            target: "_blank",
            onclick: "return false",
            href: "#"
        });
        if (r) {
            t.attr({
                href: r,
                onclick: ""
            })
        }
        j.append(s);
        j.append(t);
        A.use("honourCard", function() {});
        l(m.vstar, "vstar")
    }
    function g() {
        var r = $("<span>", {
            "class": "c-vline"
        });
        var s = $("<a>", {
            "class": "c-icon c-icon-med",
            target: "_blank",
            onclick: "return false",
            href: "#"
        });
        j.append(r);
        j.append(s);
        l(m.medical, "medical")
    }
    function i() {
        var r = $("<span>", {
            "class": "c-vline"
        });
        var s = $("<a>", {
            "class": "c-icon c-icon-air",
            target: "_blank",
            onclick: "return false",
            href: "#"
        });
        j.append(r);
        j.append(s);
        l(m.aviation, "aviation")
    }
    function l(t, w) {
        var z = t.hint;
        var y = "over";
        var s = t.url;
        if (!t ||!z) {
            return 
        }
        if (w == "vstar") {
            var x = "<div class='c-tip-cer hitcon'><ul>"
        } else {
            var x = "<div class='c-tip-info hitcon'><ul>"
        }
        for (var u = 0; u < z.length; u++) {
            if (z[u] == "") {
                y = "none";
                continue
            }
            x += "<li ";
            if (z[u].icon) {
                x += "class='c-tip-item-i'><img src='" + z[u].icon + "' class='c-customicon c-gap-icon-right-small c-tip-item-icon' />"
            } else {
                x += ">"
            }
            x += a(z[u].txt);
            x += "</li>"
        }
        x += "</ul></div>";
        var r = new bds.se.tip({
            target: j,
            mode: y,
            align: "auto",
            title: t.label + "：",
            content: x,
            offset: {
                x: 3,
                y: 25
            }
        });
        r.onShow = function() {
            var C = 1;
            for (var B = 0; B < z.length; B++) {
                if (z[B].vlevel > C) {
                    C = z[B].vlevel
                }
            }
            ns_c({
                hintUrl: j.attr("data_key"),
                hintTpl: w,
                hintId: C
            });
            if (x.indexOf("ecard")!=-1) {
                setTimeout(function() {
                    A.use("honourCard", function() {
                        var D = $(r.getDom()).find(".c-trust-ecard");
                        A.ui.honourCard(D, s, C, D.attr("value"))
                    })
                }, 0)
            }
            $("li", this.dom).each(function(D) {
                $("a", this).each(function(E) {
                    this.onmousedown = function() {
                        ns_c({
                            hintUrl: s,
                            hintTpl: w,
                            title: this.innerHTML,
                            pos: E
                        })
                    }
                })
            })
        }
    }
    function a(s) {
        var r = s;
        r = r.replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/'/g, "&#39;").replace(/"/g, "&quot;");
        r = r.replace("[/url]", "</a>").replace(/\[url ([^\s]*)\]/, "<a href='$1' target='_blank'>");
        r = r.replace(/\[img ([^\s]*)\]/, "<img src='$1' />");
        r = r.replace(/\[ecard (-?[\d]{0,3})\]/, "<div class='c-trust-ecard' value='$1'></div>");
        return r
    }
    q();
    return {
        init: q,
        render: b
    }
}();
var __callback_names = {};
function isbase64(M) {
    var o;
    if (bds && bds._base64) {
        o = bds._base64
    } else {
        o = {
            domain: bds.util.domain && bds.util.domain.get ? bds.util.domain.get("http://b1.bdstatic.com/"): "http://b1.bdstatic.com/",
            b64Exp: - 1,
            pdc: false,
            sep: 16
        };
        bds._base64 = o
    }
    var a = {
        left: "content_left",
        right: "container"
    };
    var u = o.domain;
    var z = {};
    var y = [];
    var B = {
        left: $.Deferred(),
        right: $.Deferred()
    };
    var C = {
        left: {},
        right: {}
    };
    var E = false;
    var L = 0;
    var j = 0;
    var t = 0;
    var w = null;
    var g = 1;
    o.inline = false;
    var d = [];
    function I() {
        B = {
            left: $.Deferred(),
            right: $.Deferred()
        }
    }
    M.onSendlog(function() {
        var P = [];
        if (z) {
            $.each(z, function(Q, R) {
                P.push(Q + "_" + R)
            })
        }
        M.setParam("cus_cusval", P.join("|"));
        if (H.isinline()) {
            M.setParam("cus_b64il", H.ilsum);
            if (H.ilparseTime) {
                M.setParam("cus_b64ilpt", H.ilparseTime)
            }
            if (H.ilrenderTime) {
                M.setParam("cus_b64ilrt", H.ilrenderTime)
            }
        }
    });
    function k(R, T) {
        t++;
        var R = R || [], T = T || [];
        R = $.grep(R, function(U) {
            if (C.right.hasOwnProperty(U)) {
                return false
            }
            C.right[U] = false;
            return true
        });
        T = $.grep(T, function(U) {
            if (C.left.hasOwnProperty(U)) {
                return false
            }
            C.left[U] = false;
            return true
        });
        if (o.b64Exp == 2) {
            if (T.length > 0) {
                E = true;
                p(T, "left", "reql")
            }
        }
        if (R.length > 0) {
            if (R.length > 12) {
                var S = Math.round(R.length / 2);
                var Q = [], P = [];
                $.each(R, function(U, V) {
                    U < S ? Q.push(V) : P.push(V)
                });
                p(Q, "right", "reqr2");
                p(P, "right", "reqr1")
            } else {
                p(R, "right", "reqr")
            }
        }
    }
    function J(P) {
        var R = P, Q = 0;
        while (__callback_names.hasOwnProperty(P) || window[P]) {
            P = R + "_" + Q;
            Q++
        }
        __callback_names[P] = 1;
        return P
    }
    function f(S) {
        if (typeof S == "string") {
            var Q, R = 0, P = 0;
            for (Q = 0; Q < S.length; Q++) {
                P = Q%20 + 1;
                R += S.charCodeAt(Q)<<P
            }
            return Math.abs(R)
        }
        return 0
    }
    function p(P, T, U) {
        var R = u + "image?imglist=" + P.join(",");
        var Q = f(P.join(""));
        Q = "cb_" + (Q + "").substr(Math.max(0, Q.length - 8), 8) + "_" + d.length;
        Q = J(Q);
        R += "&cb=" + Q;
        var S = new Date() * 1;
        var V = $.ajax({
            url: R,
            cache: true,
            dataType: "jsonp",
            jsonp: false,
            timeout: 1500,
            jsonpCallback: Q,
            success: function(W) {
                z[U] = new Date() * 1 - S;
                if (T == "right") {
                    F(W)
                } else {
                    if (T == "left") {
                        K(W)
                    }
                }
            }
        });
        V.always(function() {
            delete __callback_names[Q]
        });
        d.push(V)
    }
    function n() {
        var Q = d.concat(B.left, B.right);
        var P = w = $.when.apply($, Q);
        w.always(function() {
            var R =+ (new Date());
            if (P !== w) {
                return 
            }
            if (o.b64Exp == 2) {
                r("left")
            }
            r("right")
        })
    }
    var O = function(T, S, P, Q) {
        if (!Q) {
            Q = document.getElementById(a[S])
        } else {
            Q = $(Q).find("#" + a[S])[0]
        }
        if (!Q) {
            return 
        }
        var V = Q.getElementsByTagName("IMG");
        for (var R = 0; R < V.length; R++) {
            var U = V[R].getAttribute(P);
            if (U) {
                if (T.hasOwnProperty(U) && T[U]) {
                    s(V[R], T[U])
                } else {
                    x(V[R])
                }
            }
        }
        b()
    };
    var r = function(P) {
        O(C[P], P, "data-b64-id")
    };
    var m = false;
    var i = false;
    var l = function(Q, P) {
        if (!m) {
            O(Q, "right", "data-b64il-id", P)
        }
        if (P) {
            m = true
        }
        i = true
    };
    function b() {
        setTimeout(function() {
            for (var P = 0; P < y.length; P++) {
                var Q = y[P];
                if (!Q.loaded) {
                    x(Q.obj)
                }
            }
            y = []
        }, 200)
    }
    function s(P, Q) {
        try {
            P.onerror = function() {
                x(this)
            };
            y.push({
                obj: P,
                loaded: false
            });
            P.onload = function() {
                for (var T = 0; T < y.length; T++) {
                    var S = y[T];
                    if (S.obj == this) {
                        S.loaded = true
                    }
                }
            };
            P.src = "data:image/jpeg;base64," + Q
        } catch (R) {
            x(P)
        }
    }
    var x = function(P) {
        if ((P.getAttribute("data-b64-id") || P.getAttribute("data-b64il-id")) && P.getAttribute("data-src") != null) {
            P.src = P.getAttribute("data-src")
        }
    };
    var F = function(P) {
        q(P, "right")
    };
    var K = function(P) {
        q(P, "left")
    };
    var q = function(R, Q) {
        for (var P in R) {
            if (R.hasOwnProperty(P)) {
                C[Q][P] = R[P]
            }
        }
    };
    var D = function(P) {
        B[P].resolve()
    };
    var N = function() {
        C = null;
        y = null;
        B = null;
        $.each(d, function() {
            this.abort()
        })
    };
    var H = {
        loadImg: k,
        setDomLoad: D,
        end: n,
        isinline: function() {
            return i
        },
        restart: I,
        destroy: N,
        reqT: z,
        inline: l
    };
    return H
}
$(function() {
    if (bds.comm.user && bds.comm.user != "") {
        setTimeout(function() {
            $.ajax({
                dataType: "script",
                cache: true,
                url: (bds.su && bds.su.urStatic ? bds.su.urStatic : "http://su.bdimg.com") + "/static/message/js/mt_show_1.6.js",
                success: function() {
                    function a() {
                        if ($("#imsg")[0] && $("#u")[0] && $("#user")[0]) {
                            bds.se.message && bds.se.message.init && bds.se.message.init({
                                button: $("#imsg"),
                                refer: $("#u")
                            });
                            $("#user").on("mouseover", function() {
                                $("#s_mod_msg").hide()
                            })
                        }
                        if ($("#imsg1")[0] && $("#u1")[0] && $("#user1")[0]) {
                            bds.se.message && bds.se.message.init && bds.se.message.init({
                                button: $("#imsg1"),
                                refer: $("#u1")
                            });
                            $("#user1").on("mouseover", function() {
                                $("#s_mod_msg").hide()
                            })
                        }
                    }
                    function b() {
                        bds.se.message && bds.se.message.addStyle && bds.se.message.addStyle()
                    }
                    bds.comm.loginAction.push(function(d, f) {
                        if (d == 1) {
                            a();
                            b()
                        }
                    });
                    if (bds.comm.newindex) {
                        $(window).on("index_off", function() {
                            setTimeout(function() {
                                a();
                                b()
                            }, 0)
                        })
                    } else {
                        a();
                        b()
                    }
                    $(window).on("swap_end", b)
                }
            })
        }, 0)
    }
});
$(window).on("swap_end", function() {
    var f = '<div id="_FP_userDataDiv" style="behavior:url(#default#userdata);width:0px;height:0px;position:absolute;top:-10000px;left:-10000px"></div><div id="_FP_comDiv" style="behavior:url(#default#clientCaps);width:0px;height:0px;position:absolute;top:-10000px;left:-10000px"></div>';
    $("body").append(f);
    var m = "//www.baidu.com/cache/fpid/o.swf";
    var j = "//www.baidu.com/cache/fpid/lib_1_0.js";
    var k = "//www.baidu.com/cache/fpid/ielib_1_2.js";
    var b = "//www.baidu.com/cache/fpid/chromelib_1_1.js";
    var i = document.title;
    var g = {
        flashDomId: "_FP_userDataDiv",
        flashUrl: m,
        comDomId: "_FP_comDiv",
        IEStoreDomId: "_FP_userDataDiv"
    };
    var a = navigator.userAgent.toLowerCase();
    var d = false;
    if (a.indexOf("msie") >= 0 || new RegExp("trident(.*)rv.(\\d+)\\.(\\d+)").test(a)) {
        d = true
    }
    var l = function(p) {
        if (d) {
            window.setTimeout(function() {
                document.title = i
            }, 0)
        }
        window._FPID_CACHE = p;
        $("#_FP_userDataDiv").remove();
        $("#_FP_comDiv").remove();
        var t = bds.comm.qid;
        var s = "_WWW_BR_API_" + (new Date()).getTime();
        var o = window[s] = new Image();
        o.onload = function() {
            window[s] = null
        };
        var n = $.getCookie("BAIDUID");
        var r = bds && bds.util && bds.util.domain ? bds.util.domain.get("http://eclick.baidu.com/ps_fp.htm?"): "http://eclick.baidu.com/ps_fp.htm?";
        var q = r + "pid=ps&fp=" + p.data.fp + "&im=" + p.data.im + "&wf=" + p.data.wf + "&br=" + p.data.br + "&qid=" + t + "&bi=" + n;
        o.src = q
    };
    if (window._FPID_CACHE) {
        window._FPIDTimer = window.setTimeout(function() {
            l(window._FPID_CACHE)
        }, 2500);
        return 
    }
    window._FPIDTimer = window.setTimeout(function() {
        var n = "";
        if (d) {
            n = k
        } else {
            n = b
        }
        $.ajax({
            url: n,
            cache: true,
            dataType: "script",
            success: function() {
                fpLib.getFp(l, g)
            }
        })
    }, 2500)
});
$(window).on("swap_begin", function() {
    if (window._FPIDTimer) {
        window.clearTimeout(window._FPIDTimer);
        $("#_FP_userDataDiv").remove();
        $("#_FP_comDiv").remove()
    }
});
var bds = bds || {};
bds.se = bds.se || {};
bds.se.upn = {
    regexp: /BD_UPN=([\w|\d]*)/,
    cookieset: [],
    write: function(a) {
        document.cookie = "BD_UPN=" + a + "; expires=" + (new Date(new Date().getTime() + 864000000)).toGMTString()
    },
    set: function(a) {
        var b = this;
        try {
            if ($.isArray(a)) {
                b.cookieset = b.cookieset.concat(a)
            }
        } catch (d) {}
    },
    run: function() {
        var f = this;
        try {
            var g = "";
            for (var d = 0; d < f.cookieset.length; d++) {
                if (f.cookieset[d] && f.cookieset[d].k && f.cookieset[d].v) {
                    var b = f.cookieset[d].k + "";
                    var a = f.cookieset[d].v + "";
                    if (b.length == a.length == 1) {
                        var l = {};
                        l[b] = a;
                        g = g + b + a
                    }
                }
            }
            f.write(g)
        } catch (j) {}
    }
};
bds.se.upn.set((function() {
    var a = navigator.userAgent;
    var g = a.toLowerCase();
    function m() {
        if (g.indexOf("lbbrowser") > 0) {
            return g.match(/lbbrowser/gi)
        }
        if (g.indexOf("maxthon") > 0) {
            return g.match(/maxthon\/[\d.]+/gi)
        }
        if (g.indexOf("bidubrowser") > 0) {
            return g.match(/bidubrowser/gi)
        }
        if (g.indexOf("baiduclient") > 0) {
            return g.match(/baiduclient/gi)
        }
        if (g.indexOf("metasr") > 0) {
            return g.match(/metasr/gi)
        }
        if (g.indexOf("qqbrowser") > 0) {
            return g.match(/qqbrowser/gi)
        }
        if (!(function() {
            if (navigator.mimeTypes.length > 0) {
                var b;
                for (b in navigator.mimeTypes) {
                    if (navigator.mimeTypes[b]["type"] == "application/vnd.chromium.remoting-viewer") {
                        return true
                    }
                }
            }
            return false
        })() && (("track" in document.createElement("track"))&&!("scoped" in document.createElement("style"))&&!("v8Locale" in window) && /Gecko\)\s+Chrome/.test(navigator.appVersion)) || (("track" in document.createElement("track")) && ("scoped" in document.createElement("style")) && ("v8Locale" in window))) {
            return "qihu"
        }
        if (g.indexOf("msie") > 0) {
            return g.match(/msie [\d.]+;/gi)
        }
        if (window.document.documentMode) {
            return "msie"
        }
        if (g.indexOf("firefox") > 0) {
            return g.match(/firefox\/[\d.]+/gi)
        }
        if (g.indexOf("opr") > 0) {
            return g.match(/opr\/[\d.]+/gi)
        }
        if (g.indexOf("chrome") > 0) {
            return g.match(/chrome\/[\d.]+/gi)
        }
        if (g.indexOf("safari") > 0 && g.indexOf("chrome") < 0) {
            return g.match(/safari\/[\d.]+/gi)
        }
        return ""
    }
    browser = (m() + "").replace(/[0-9.\/|;|\s]/ig, "");
    browserversion = (function() {
        if (browser == "msie") {
            if (a.search(/MSIE [2-5]/) > 0) {
                return "ie5"
            }
            if (a.indexOf("MSIE 6") > 0) {
                return "ie6"
            }
            if (a.indexOf("MSIE 7") > 0) {
                return "ie7"
            }
            if (a.indexOf("MSIE 8") > 0) {
                return "ie8"
            }
            if (a.indexOf("MSIE 9") > 0) {
                return "ie9"
            }
            if (a.indexOf("MSIE 10") > 0) {
                return "ie10"
            }
            if (window.document.documentMode == "11") {
                return "ie11"
            }
            return "other"
        } else {
            return ""
        }
    })();
    browsertype = (function() {
        if (g.indexOf("msie") > 0 || new RegExp("trident(.*)rv.(\\d+)\\.(\\d+)").test(g)) {
            return "ie"
        }
        if (g.indexOf("firefox") > 0) {
            return "firefox"
        }
        if (g.indexOf("chrome") > 0) {
            return "chrome"
        }
        if (g.indexOf("safari") > 0 && g.indexOf("chrome") < 0) {
            return "safari"
        }
        return "other"
    })();
    function l() {
        var n = (navigator.platform == "Win32") || (navigator.platform == "Windows");
        var o = (navigator.platform == "Mac68K") || (navigator.platform == "MacPPC") || (navigator.platform == "Macintosh") || (navigator.platform == "MacIntel");
        if (o) {
            return "mac"
        }
        var b = (navigator.platform == "X11")&&!n&&!o;
        if (b) {
            return "unix"
        }
        var p = (String(navigator.platform).indexOf("Linux")>-1);
        if (p) {
            return "linux"
        }
        if (n) {
            return "windows"
        }
        return "other"
    }
    os = l();
    osversion = (function() {
        if (os == "windows") {
            if (a.indexOf("Windows NT 5.1")>-1 || a.indexOf("Windows XP")>-1) {
                return "xp"
            }
            if (isWinVista = a.indexOf("Windows NT 6.0")>-1 || a.indexOf("Windows Vista")>-1) {
                return "vista"
            }
            if (a.indexOf("Windows NT 6.1")>-1 || a.indexOf("Windows 7")>-1) {
                return "win7"
            }
            if (a.indexOf("Windows NT 6.2")>-1 || a.indexOf("Windows 8")>-1) {
                return "win8"
            }
            if (a.indexOf("Windows NT 6.3")>-1 || a.indexOf("Windows 8.1")>-1) {
                return "win8.1"
            }
            return "other"
        }
    })();
    var i = (function(n) {
        var b = 0;
        switch (n) {
        case"msie":
            b = 1;
            break;
        case"chrome":
            b = 2;
            break;
        case"firefox":
            b = 3;
            break;
        case"safari":
            b = 4;
            break;
        case"opr":
            b = 5;
            break;
        case"lbbrowser":
            b = 6;
            break;
        case"maxthon":
            b = 7;
            break;
        case"bidubrowser":
            b = 8;
            break;
        case"metasr":
            b = 9;
            break;
        case"qqbrowser":
            b = "a";
            break;
        case"qihu":
            b = "b";
            break;
        case"baiduclient":
            b = "c";
            break
        }
        return b
    })(browser);
    var j = (function(n) {
        var b = 0;
        switch (n) {
        case"ie6":
            b = 1;
            break;
        case"ie7":
            b = 2;
            break;
        case"ie8":
            b = 3;
            break;
        case"ie9":
            b = 4;
            break;
        case"ie10":
            b = 5;
            break;
        case"ie11":
            b = 6;
            break;
        case"other":
            b = 7;
            break
        }
        return b
    })(browserversion);
    var d = (function(n) {
        var b = 0;
        switch (n) {
        case"windows":
            b = 1;
            break;
        case"mac":
            b = 2;
            break;
        case"linux":
            b = 3;
            break;
        case"unix":
            b = 4;
            break
        }
        return b
    })(os);
    var f = (function(n) {
        var b = 0;
        switch (n) {
        case"xp":
            b = 1;
            break;
        case"vista":
            b = 2;
            break;
        case"win7":
            b = 3;
            break;
        case"win8":
            b = 4;
            break;
        case"win8.1":
            b = 5;
            break;
        case"other":
            b = 6;
            break
        }
        return b
    })(osversion);
    var k = (function(n) {
        var b = 0;
        switch (n) {
        case"ie":
            b = 1;
            break;
        case"firefox":
            b = 2;
            break;
        case"chrome":
            b = 3;
            break;
        case"safari":
            b = 4;
            break
        }
        return b
    })(browsertype);
    return [{
        k: 1,
        v: i
    }, {
        k: 2,
        v: j
    }, {
        k: 3,
        v: d
    }, {
        k: 4,
        v: f
    }, {
        k: 5,
        v: k
    }
    ]
})());
bds.se.heightControl = {
    check: function() {
        return $("#content_right").height() > $("#content_left").height()
    },
    cleanEC: function() {
        var d = $(".ec_bdtg"), b = $("#ec_im_container").children("div"), g = b.length, f = g - 1;
        if (bds.se.heightControl.check()) {
            if (d && d.length) {
                d.css("display", "none")
            }
        }
        while (bds.se.heightControl.check() && f >= 0) {
            var a = b[f];
            $(a).css("display", "none");
            f--
        }
    },
    cleanRes: function() {
        var g = $("#content_right").find(".result-op"), b = g.length, a = b - 1;
        if (a == 0) {
            var f = $(g[0]).parent();
            if (($("#content_right").height() + g.height()) < $("#content_left").height()) {
                f.css({
                    position: "static"
                })
            }
        } else {
            while (bds.se.heightControl.check() && a > 0) {
                var d = g[a];
                $(d).css("display", "none");
                a--
            }
        }
    },
    init: function() {
        bds.se.heightControl.cleanEC();
        bds.se.heightControl.cleanRes()
    }
};
$(window).on("swap_end", function() {
    if (bds.comm.encTn) {
        $.setCookie("H_PS_645EC", bds.comm.encTn, {
            expires: 24 * 60 * 60 * 30
        })
    }
    if (bds.se.trust) {
        bds.se.trust.init()
    }
    $imgs = $("#content_left .c-container img[data-zoomImg]:not([data-zoomImg=''])");
    if ($imgs.length > 0 && bds.comm.imgZoomHover) {
        require(["ui/ImgZoomHover"], function(d) {
            var f = new d();
            f.render($imgs)
        });
        bds.comm.imgZoomHover = false
    } else {
        if ($imgs.length > 0 && bds.comm.imgZoomHover1) {
            require(["ui/ImgZoomHover1"], function(d) {
                var f = new d();
                f.render($imgs)
            });
            bds.comm.imgZoomHover1 = false
        } else {
            if ($imgs.length > 0 && bds.comm.imgZoomHover2) {
                require(["ui/ImgZoomHover2"], function(d) {
                    var f = new d();
                    f.render($imgs)
                });
                bds.comm.imgZoomHover2 = false
            } else {
                if ($imgs.length > 0 && bds.comm.imgZoomHover3) {
                    require(["ui/ImgZoomHover3"], function(d) {
                        var f = new d();
                        f.render($imgs)
                    });
                    bds.comm.imgZoomHover3 = false
                }
            }
        }
    }
    bds.se.heightControl.init();
    bds.util.setContainerWidth();
    $(document).delegate(".feedback", "click", function() {
        var d = this;
        $.getScript("http://s1.bdstatic.com/r/www/cache/static/plugins/feedback_d057de2a.js", function() {
            var i = d.getAttribute("data-feedbackid") || 1;
            var g = {
                product_id: 18,
                entrance_id: i,
                needGuide: true,
                showPosition: "right"
            };
            var f = {
                username: bds.comm.username,
                query: bds.comm.query,
                fb_qid: bds.comm.qid
            };
            bds.qa.ShortCut.initRightBar(g, f)
        })
    });
    var a = $("#form").find('input[type="hidden"][name=rsv_pq]');
    if (a.length) {
        $(a).val(bds.comm.qid)
    } else {
        $("#form").append('<input type="hidden" name="rsv_pq" value="' + bds.comm.qid + '"/>')
    }
    var b = $("#form").find('input[type="hidden"][name=rsv_t]');
    if (b.length) {
        $(b).val(bds.comm.encTn)
    } else {
        $("#form").append('<input type="hidden" name="rsv_t" value="' + bds.comm.encTn + '"/>')
    }
});
(function() {
    $(window).one("swap_end", function() {
        $("body").on("mousedown", ".se_common_hint a", function() {
            var a = $(this), f = a.parents(".se_common_hint"), g = f.attr("data-id") || "", d = f.attr("data-tpl") || "", b = f.find("a").index(a);
            ns_c_pj({
                hintId: g,
                hintTpl: d,
                title: a.html(),
                pos: b,
                qid: bds.comm.qid || ""
            }, "pj=hint&")
        })
    })
})();
$(function() {
    $("#u,#u1").delegate(".lb", "click", function() {
        try {
            bds.se.login.open()
        } catch (a) {}
    })
});
$.ajax({
    dataType: "script",
    cache: true,
    url: "http://s1.bdstatic.com/r/www/cache/static/home/js/nu_instant_search_314270a9.js"
});
(function() {
    var b = new Date() * 1;
    var a = function(f, g) {
        f = bds.util.domain && bds.util.domain.get ? bds.util.domain.get(f) : f;
        var i = new Image();
        i.src = f + "?_t=" + (g ? g : b);
        i.onload = (i.onerror = function() {
            i = null
        })
    };
    try {
        if (!window.pageState || window.pageState == 0) {
            $("#kw1,#kw").one("keydown", function() {
                a("http://b1.bdstatic.com/img/pc.gif", parseInt(Math.random() * 1000));
                a("http://ecma.bdimg.com/public03/pc.gif");
                $.each(["i7", "i8", "i9", "t10", "t11", "t12"], function(f, g) {
                    a("http://" + g + ".baidu.com/ps_default.gif")
                })
            })
        }
    } catch (d) {}
})();
(function() {
    $.ajaxPrefilter("parts", function(b, a, d) {
        b.__partsCallback = [];
        b.__partsIndex = 0;
        d.parts = function(f) {
            b.__partsCallback.push(f)
        };
        if (b.parts) {
            d.parts(b.parts)
        }
        b.converters["* parts"] = function(f) {
            return f
        }
    });
    $.ajaxTransport("parts", function(a) {
        if (!a.crossDomain || support.cors) {
            var b;
            return {
                send: function(j, d) {
                    var f, g = a.xhr();
                    g.open(a.type, a.url, a.async, a.username, a.password);
                    if (a.xhrFields) {
                        for (f in a.xhrFields) {
                            g[f] = a.xhrFields[f]
                        }
                    }
                    if (a.mimeType && g.overrideMimeType) {
                        g.overrideMimeType(a.mimeType)
                    }
                    if (!a.crossDomain&&!j["X-Requested-With"]) {
                        j["X-Requested-With"] = "XMLHttpRequest"
                    }
                    for (f in j) {
                        if (j[f] !== undefined) {
                            g.setRequestHeader(f, j[f] + "")
                        }
                    }
                    g.send((a.hasContent && a.data) || null);
                    b = function(m, l) {
                        var k, q, o, n;
                        if ((g.readyState === 3 || g.readyState === 4)&&!l) {
                            (function() {
                                var r = a.delimiter;
                                var w = "";
                                try {
                                    w = g.responseText
                                } catch (x) {}
                                if (w == "") {
                                    return 
                                }
                                var y =- 1, u, t = 0, s;
                                if (r) {
                                    while (true) {
                                        for (; t <= a.__partsIndex; t++) {
                                            u = (y==-1) ? 0 : y + r.length;
                                            y = w.indexOf(r, u);
                                            if (y==-1) {
                                                break
                                            }
                                        }
                                        if (y==-1 && g.readyState !== 4) {
                                            return 
                                        }
                                        for (s = 0; s < a.__partsCallback.length; s++) {
                                            a.__partsCallback[s].call(g, w.substring(u, y==-1 ? w.length : y), a.__partsIndex, w)
                                        }
                                        a.__partsIndex++;
                                        if (y==-1) {
                                            return 
                                        }
                                    }
                                } else {
                                    for (t = 0; t < a.__partsCallback.length; t++) {
                                        a.__partsCallback[t].call(g, w)
                                    }
                                }
                            })()
                        }
                        if (b && (l || g.readyState === 4)) {
                            b = undefined;
                            g.onreadystatechange = jQuery.noop;
                            if (l) {
                                if (g.readyState !== 4) {
                                    g.abort()
                                }
                            } else {
                                o = {};
                                k = g.status;
                                if (typeof g.responseText === "string") {
                                    o.text = g.responseText
                                }
                                try {
                                    q = g.statusText
                                } catch (p) {
                                    q = ""
                                }
                                if (!k && a.isLocal&&!a.crossDomain) {
                                    k = o.text ? 200 : 404
                                } else {
                                    if (k === 1223) {
                                        k = 204
                                    }
                                }
                            }
                        }
                        if (o) {
                            d(k, q, o, g.getAllResponseHeaders())
                        }
                    };
                    if (!a.async) {
                        b()
                    } else {
                        if (g.readyState === 4) {
                            setTimeout(b)
                        } else {
                            g.onreadystatechange = b
                        }
                    }
                },
                abort: function() {
                    if (b) {
                        b(undefined, true)
                    }
                }
            }
        }
    })
})();
(function() {
    var defaultOptions = {
        sugSet: 1,
        sugStoreSet: 1,
        isSwitch: 1,
        imeSwitch: 0,
        resultNum: 10,
        resultLang: 0
    }, options = {}, tmpName;
    var expire30y = new Date();
    expire30y.setTime(expire30y.getTime() + 30 * 365 * 86400000);
    try {
        if (bds && bds.comm && bds.comm.personalData) {
            if (typeof bds.comm.personalData == "string") {
                bds.comm.personalData = eval("(" + bds.comm.personalData + ")")
            }
            if (!bds.comm.personalData) {
                return 
            }
            for (tmpName in bds.comm.personalData) {
                if (defaultOptions.hasOwnProperty(tmpName) && bds.comm.personalData.hasOwnProperty(tmpName)) {
                    if (bds.comm.personalData[tmpName].ErrMsg == "SUCCESS") {
                        options[tmpName] = bds.comm.personalData[tmpName].value
                    }
                }
            }
        }
        try {
            if (!parseInt(options.resultNum)) {
                delete (options.resultNum)
            }
            if (!parseInt(options.resultLang) && options.resultLang != "0") {
                delete (options.resultLang)
            }
        } catch (e) {}
        writeCookie();
        if (!("sugSet" in options)) {
            options.sugSet = (Cookie.get("sug", 3) != 3 ? 0 : 1)
        }
        if (!("sugStoreSet" in options)) {
            options.sugStoreSet = Cookie.get("sugstore", 1)
        }
        var BAIDUID = Cookie.get("BAIDUID");
        if (!("resultNum" in options)) {
            if (/NR=(\d+)/.test(BAIDUID)) {
                options.resultNum = RegExp.$1 ? parseInt(RegExp.$1) : 10
            }
        }
        if (!("resultLang" in options)) {
            if (/SL=(\d+)/.test(BAIDUID)) {
                options.resultLang = RegExp.$1 ? parseInt(RegExp.$1) : 0
            }
        }
        if (!("isSwitch" in options)) {
            options.isSwitch = (Cookie.get("ORIGIN", 0) == 2 ? 0 : 1)
        }
        if (!("imeSwitch" in options)) {
            options.imeSwitch = Cookie.get("bdime", 0)
        }
    } catch (e) {}
    function save(callback) {
        var optionsStr = [];
        for (tmpName in options) {
            if (options.hasOwnProperty(tmpName)) {
                optionsStr.push('"' + tmpName + '":"' + options[tmpName] + '"')
            }
        }
        var str = "{" + optionsStr.join(",") + "}";
        if (bds.comm.personalData) {
            $.ajax({
                url: "//www.baidu.com/ups/submit/addtips/?product=ps&tips=" + encodeURIComponent(str) + "&_r=" + new Date().getTime(),
                success: function() {
                    writeCookie();
                    if (typeof callback == "function") {
                        callback()
                    }
                }
            })
        } else {
            writeCookie();
            if (typeof callback == "function") {
                setTimeout(callback, 0)
            }
        }
    }
    function set(optionName, value) {
        options[optionName] = value
    }
    function get(optionName) {
        return options[optionName]
    }
    function writeCookie() {
        if (options.hasOwnProperty("sugSet")) {
            var value = options.sugSet == "0" ? "0": "3";
            clearCookie("sug");
            Cookie.set("sug", value, document.domain, "/", expire30y)
        }
        if (options.hasOwnProperty("sugStoreSet")) {
            var value = options.sugStoreSet == 0 ? "0": "1";
            clearCookie("sugstore");
            Cookie.set("sugstore", value, document.domain, "/", expire30y)
        }
        if (options.hasOwnProperty("isSwitch")) {
            var value = options.isSwitch == 0 ? "2": "0";
            clearCookie("ORIGIN");
            Cookie.set("ORIGIN", value, document.domain, "/", expire30y)
        }
        if (options.hasOwnProperty("imeSwitch")) {
            var value = options.imeSwitch;
            clearCookie("bdime");
            Cookie.set("bdime", value, document.domain, "/", expire30y)
        }
    }
    function writeBAIDUID() {
        var BAIDUID = Cookie.get("BAIDUID"), NR, FG, SL;
        if (/FG=(\d+)/.test(BAIDUID)) {
            FG = RegExp.$1
        }
        if (/SL=(\d+)/.test(BAIDUID)) {
            SL = RegExp.$1
        }
        if (/NR=(\d+)/.test(BAIDUID)) {
            NR = RegExp.$1
        }
        if (options.hasOwnProperty("resultNum")) {
            NR = options.resultNum
        }
        if (options.hasOwnProperty("resultLang")) {
            SL = options.resultLang
        }
        Cookie.set("BAIDUID", BAIDUID.replace(/:.*$/, "") + (typeof SL != "undefined" ? ":SL=" + SL : "") + (typeof NR != "undefined" ? ":NR=" + NR : "") + (typeof FG != "undefined" ? ":FG=" + FG : ""), ".baidu.com", "/", expire30y, true)
    }
    function clearCookie(name) {
        Cookie.clear(name, "/");
        Cookie.clear(name, "/", document.domain);
        Cookie.clear(name, "/", "." + document.domain);
        Cookie.clear(name, "/", ".baidu.com")
    }
    function reset(callback) {
        options = defaultOptions;
        save(callback)
    }
    window.UPS = {
        writeBAIDUID: writeBAIDUID,
        reset: reset,
        get: get,
        set: set,
        save: save
    }
})();
$(function() {
    $("#head").delegate(".index_tab_top>a,.index_tab_bottom>a,#u1>a", "mousedown", function() {
        return ns_c({
            fm: "behs",
            tab: $(this).attr("name"),
            query: "",
            un: encodeURIComponent(bds.comm.user || "")
        })
    })
});
if (bds.comm.resultPage) {} else {
    if (bds.se.upn) {
        setTimeout(function() {
            bds.se.upn.run()
        }, 3000)
    }
};
