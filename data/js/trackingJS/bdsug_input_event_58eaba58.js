(function() {
    var bds = window.bds || {};
    bds.se = bds.se || {};
    bds.se.store = (function() {
        var l = {}, h = window, k = h.document, c = "localStorage", o = "globalStorage", d = "__storejs__", g;
        l.disabled = false;
        l.set = function(e, p) {};
        l.get = function(e) {};
        l.remove = function(e) {};
        l.clear = function() {};
        l.transact = function(e, r, p) {
            var q = l.get(e);
            if (p == null) {
                p = r;
                r = null
            }
            if (typeof q == "undefined") {
                q = r || {}
            }
            p(q);
            l.set(e, q)
        };
        l.getAll = function() {};
        l.serialize = function(e) {
            return String(e)
        };
        l.deserialize = function(e) {
            if (typeof e != "string") {
                return undefined
            }
            return e
        };
        function b() {
            try {
                return (c in h && h[c])
            } catch (e) {
                return false
            }
        }
        function n() {
            try {
                return (o in h && h[o] && h[o][h.location.hostname])
            } catch (e) {
                return false
            }
        }
        if (b()) {
            g = h[c];
            l.set = function(e, p) {
                if (p === undefined) {
                    return l.remove(e)
                }
                g.setItem(e, l.serialize(p))
            };
            l.get = function(e) {
                return l.deserialize(g.getItem(e))
            };
            l.remove = function(e) {
                g.removeItem(e)
            };
            l.clear = function() {
                g.clear()
            };
            l.getAll = function() {
                var e = {};
                for (var q = 0; q < g.length; ++q) {
                    var p = g.key(q);
                    e[p] = l.get(p)
                }
                return e
            }
        } else {
            if (n()) {
                g = h[o][h.location.hostname];
                l.set = function(e, p) {
                    if (p === undefined) {
                        return l.remove(e)
                    }
                    g[e] = l.serialize(p)
                };
                l.get = function(e) {
                    return l.deserialize(g[e] && g[e].value)
                };
                l.remove = function(e) {
                    delete g[e]
                };
                l.clear = function() {
                    for (var e in g) {
                        delete g[e]
                    }
                };
                l.getAll = function() {
                    var e = {};
                    for (var q = 0; q < g.length; ++q) {
                        var p = g.key(q);
                        e[p] = l.get(p)
                    }
                    return e
                }
            } else {
                if (k.documentElement.addBehavior) {
                    var j, f;
                    try {
                        f = new ActiveXObject("htmlfile");
                        f.open();
                        f.write("<script>document.w=window<\/script><iframe src='/favicon.ico'></iframe>");
                        f.close();
                        j = f.w.frames[0].document;
                        g = j.createElement("div")
                    } catch (i) {
                        g = k.createElement("div");
                        j = k.body
                    }
                    function a(e) {
                        return function() {
                            var q = Array.prototype.slice.call(arguments, 0);
                            q.unshift(g);
                            j.appendChild(g);
                            g.addBehavior("#default#userData");
                            g.load(c);
                            var p = e.apply(l, q);
                            j.removeChild(g);
                            return p
                        }
                    }
                    function m(e) {
                        return "_" + e
                    }
                    l.set = a(function(q, e, p) {
                        e = m(e);
                        if (p === undefined) {
                            return l.remove(e)
                        }
                        q.setAttribute(e, l.serialize(p));
                        q.save(c)
                    });
                    l.get = a(function(p, e) {
                        e = m(e);
                        return l.deserialize(p.getAttribute(e))
                    });
                    l.remove = a(function(p, e) {
                        e = m(e);
                        p.removeAttribute(e);
                        p.save(c)
                    });
                    l.clear = a(function(r) {
                        var p = r.XMLDocument.documentElement.attributes;
                        r.load(c);
                        for (var q = 0, e; e = p[q]; q++) {
                            r.removeAttribute(e.name)
                        }
                        r.save(c)
                    });
                    l.getAll = a(function(s) {
                        var p = s.XMLDocument.documentElement.attributes;
                        s.load(c);
                        var q = {};
                        for (var r = 0, e; e = p[r]; ++r) {
                            q[e] = l.get(e)
                        }
                        return q
                    })
                }
            }
        }
        try {
            l.set(d, d);
            if (l.get(d) != d) {
                l.disabled = true
            }
            l.remove(d)
        } catch (i) {
            l.disabled = true
        }
        return l
    })();
    function checkHsugIn(word) {
        if (window.__sample_hsug_length) {
            return word.length >= 4 || encodeURIComponent(word).length >= 18
        }
        return word.length >= 6 || encodeURIComponent(word).length >= 27
    }
    function checkHsugShow(word) {
        return word.length > 1 && encodeURIComponent(word).length > 3
    }
    bds.se.sug = function(opts) {
        var sug = new bdsug(opts);
        return sug.outInterface()
    };
    var supportInputEvent = "oninput" in document.createElement("input")&&!navigator.userAgent.match(/MSIE 9/)&&!navigator.userAgent.match(/chrome\/(28|29|30)/i);
    function bdsug(opts) {
        var me = this, opts = me.opts = opts || {};
        me.ipt = opts.ipt || null;
        me.reverse = opts.reverse || false;
        me.form = opts.form || null;
        me.submission = opts.submission || null;
        me.maxNum = opts.maxNum || 10;
        me.withoutMode = opts.withoutMode || false;
        me.withoutRich = opts.withoutRich || false;
        me.withoutStat = opts.withoutStat || false;
        me.withoutZhixin = opts.withoutZhixin || false;
        me.visible = false;
        me.renderCallback = opts.renderCallback || function() {};
        me.selectCallback = opts.selectCallback || function() {};
        me.storestr = me.storestr || "";
        me.storearr = me.storearr || [];
        me.zhixinsug = [];
        me.zhixintemplate = {};
        me.zhixinused = {};
        me.zhixindata = {};
        var F = F || {
            uri: function(url) {
                return url
            }
        };
        me.zhixintemplate.tvsingle = "http://s1.bdstatic.com/r/www/cache/static/sug/js/zhixin/tvsingle_625a9dba.js";
        me.query = me.ipt && me.ipt.value || "";
        me.inputValue = me.query;
        me.showValue = me.query;
        me.sugValue = "";
        me.queryValue = "";
        me.reqValue = "";
        me.value = me.query;
        me.index =- 1;
        me.sugcontainer;
        me.dataCached = {};
        me.dataArray = [];
        me.dataStored = [];
        me.dataAladdin = [];
        me.timer;
        me.rsv_sug = 0;
        me.rsv_sug1 = 0;
        me.rsv_sug3 = 0;
        me.rsv_sug4 = 0;
        me.initTime = 0;
        me.inputT = 0;
        me.jqXhr = null;
        me.ipt && me.init()
    }
    bdsug.prototype = {
        updateInitData: function() {
            var me = this, opts = me.opts || {};
            me.setsug = true;
            me.setsugstorage = true;
            me.sets = {};
            me.sugcookie = navigator.cookieEnabled && /sug=(\d)/.test(document.cookie) ? RegExp.$1 : 3;
            me.sugstorecookie = navigator.cookieEnabled && /sugstore=(\d)/.test(document.cookie) ? RegExp.$1 : 1;
            if (bds.comm && bds.comm.personalData) {
                if (typeof bds.comm.personalData == "string") {
                    me.sets = eval("(" + bds.comm.personalData + ")")
                } else {
                    me.sets = bds.comm.personalData
                }
            }
            if (me.sets.errno && me.sets.errno == 0 && me.sets.sugSet && me.sets.sugSet.ErrMsg && me.sets.sugSet.ErrMsg == "SUCCESS") {
                if (me.sets.sugSet.value == 0) {
                    me.setsug = false
                }
            } else {
                if (me.sugcookie == 0) {
                    me.setsug = false
                }
            }
            if (me.sets.errno && me.sets.errno == 0 && me.sets.sugStoreSet && me.sets.sugStoreSet.ErrMsg && me.sets.sugStoreSet.ErrMsg == "SUCCESS") {
                if (me.sets.sugStoreSet.value == 0) {
                    me.setsugstorage = false
                }
            } else {
                if (me.sugstorecookie == 0) {
                    me.setsugstorage = false
                }
            }
            me.loggedon = bds.comm && bds.se.store&&!bds.se.store.disabled && navigator.cookieEnabled;
            me.showsug = opts.showsug ? opts.showsug : me.setsug;
            me.showsugstore = opts.showsugstore ? opts.showsugstore : (me.showsug && me.loggedon && me.setsugstorage);
            me.query = bds.comm.query;
            me.pinyin = bds.comm.pinyin;
            me.sugHost = bds.comm.sugHost || "";
            me.sid = navigator.cookieEnabled && /H_PS_PSSID=([0-9_]+)/.test(document.cookie) ? RegExp.$1 : bds.comm.sid;
            me.writeStore()
        },
        check: function() {
            var me = this;
            if (me.value != me.ipt.value && me.showValue != me.ipt.value) {
                me.inputValue = me.showValue = me.value = me.ipt.value;
                $(me.ipt).trigger("inputChange", [me]);
                me.request(me.value)
            }
        },
        startCircle: function() {
            var me = this;
            if (me.timer) {
                return 
            }
            $(me.ipt).trigger("start", [me]);
            me.timer = setTimeout(function() {
                me.check();
                me.timer = setTimeout(arguments.callee, 200)
            }, 200);
            if (supportInputEvent) {
                $(me.ipt).bind("input", function() {
                    me.check()
                })
            }
        },
        stopCircle: function() {
            var me = this;
            if (me.timer) {
                clearTimeout(me.timer);
                if (supportInputEvent) {
                    $(me.ipt).unbind("input")
                }
                me.timer = null;
                $(me.ipt).trigger("stop", [me])
            }
        },
        callback: function(data) {
            var me = this;
            if (data && data.s && (data.s.length > 0 || data.z)) {
                me.queryValue = data.q;
                if (data.q) {
                    me.dataCached[data.q] = data
                }
                if (!$(me.form).hasClass("showsugzhixin")) {
                    me.withoutZhixin = true
                }
                if (data && data.exp) {
                    me.addStat("rsv_sug6", data.exp)
                }
                if (!me.withoutZhixin) {
                    function useTemplate(type) {
                        if (me.zhixintemplate[type]&&!me.zhixinused[type]) {
                            me.zhixinused[type] = $.ajax({
                                crossDomain: true,
                                url: me.zhixintemplate[type],
                                dataType: "script",
                                scriptCharset: "UTF-8"
                            })
                        }
                    }
                    if (data.zzx) {
                        for (var i = 0; i < data.zzx.length; i++) {
                            if (data.zzx[i] && data.zzx[i].type) {
                                me.zhixinsug.push({
                                    value: data.s[i],
                                    type: data.zzx[i].type,
                                    url: data.zzx[i].url
                                });
                                useTemplate(data.zzx[i].type)
                            }
                        }
                    }
                }
                me.dataArray = me.packData(data);
                me.render(me.dataArray)
            } else {
                me.hideSug()
            }
        },
        buildUrl: function(val, mode, isPreReq) {
            var me = this;
            var requestUrl = me.sugHost || "http://suggestion.baidu.com/su";
            var sugMode = mode ? "&sugmode=" + mode: "";
            var zxMode = me.withoutZhixin ? "": "&zxmode=1";
            if (me.sugcookie > 0) {
                me.sugcookie = 3
            }
            var url = requestUrl + "?wd=" + encodeURIComponent(val) + sugMode + zxMode + "&json=1&p=" + me.sugcookie + "&sid=" + me.sid + (bds.comm.supportis ? "&req=2" : "");
            if (window.bds && bds.comm && bds.comm.cur_query) {
                url += "&bs=" + encodeURIComponent(bds.comm.cur_query);
                if (!val&&!isPreReq) {
                    url += "&sc=eb"
                }
            }
            if (window.bds && bds.comm && bds.comm.cur_disp_query) {
                url += "&pbs=" + encodeURIComponent(bds.comm.cur_disp_query)
            }
            return url
        },
        request: function(val, mode) {
            var me = this;
            if (me.showsug) {
                me.reqValue = val;
                if (me.dataCached[val]) {
                    me.callback(me.dataCached[val]);
                    return 
                }
                if (me.jqXhr) {
                    me.jqXhr.abort()
                }
                me.jqXhr = $.ajax({
                    dataType: "jsonp",
                    async: true,
                    scriptCharset: "gbk",
                    url: me.buildUrl(val, mode),
                    jsonp: "cb",
                    timeout: 5000,
                    success: function(data) {
                        me.rsv_sug4 += new Date().getTime() - me.initTime;
                        me.addStat("rsv_sug4", me.rsv_sug4);
                        me.callback(data)
                    },
                    always: function() {
                        me.jqXhr = null
                    }
                });
                me.rsv_sug3++;
                me.addStat("rsv_sug3", me.rsv_sug3);
                me.initTime = new Date().getTime()
            } else {}
        },
        packData: function(data) {
            var me = this;
            var arr = [];
            me.checkAla(data);
            me.checkStore();
            var dataMerged = me.mergeData(data);
            for (var i = 0; i < dataMerged.length; i++) {
                if (i > me.maxNum - 1) {
                    break
                }
                if (me.reverse) {
                    arr.unshift(dataMerged[i])
                } else {
                    arr.push(dataMerged[i])
                }
            }
            return arr
        },
        checkAla: function(data) {
            var me = this;
            me.dataAladdin = [];
            if (data.z && data.z.length > 0&&!me.withoutRich) {
                for (var i = 0;
                i < data.z.length; i++) {
                    var d = data.z[i];
                    if (d.id && d.type && d.key && d.word) {
                        me.dataAladdin.push({
                            value: d.key,
                            otherData: d
                        })
                    }
                }
            } else {
                me.dataAladdin = []
            }
        },
        writeStore: function(pinyin, query) {
            var me = this;
            if (me.showsugstore && me.query && me.pinyin && checkHsugIn(me.query)) {
                me.getStore();
                var q = encodeURIComponent(me.query.toLowerCase());
                var p = encodeURIComponent(me.pinyin.toLowerCase());
                var duplicate = false;
                $.each(me.storearr, function(i, a) {
                    if (a.p == p) {
                        a.q = q;
                        duplicate = true
                    }
                });
                if (!duplicate) {
                    me.storearr.push({
                        q: q,
                        p: p
                    })
                }
                if (me.storearr.length > 50) {
                    me.storearr.shift()
                }
                me.setStore()
            }
        },
        checkStore: function() {
            var me = this;
            me.dataStored = [];
            if (me.showsugstore && checkHsugShow(me.value)) {
                me.getStore();
                $.each(me.storearr, function(i, a) {
                    var q = decodeURIComponent(a.q);
                    var p = decodeURIComponent(a.p);
                    if (q.indexOf(me.value.toLowerCase()) == 0 || p.indexOf(me.value.toLowerCase()) == 0) {
                        me.dataStored.unshift({
                            value: q,
                            pinyin: p
                        })
                    }
                })
            }
            return me.dataStored
        },
        getStore: function() {
            var me = this;
            me.storestr = bds.se.store.get("BDSUGSTORED");
            me.storearr = (me.storestr && $.parseJSON(me.storestr)) || []
        },
        setStore: function() {
            var me = this;
            var str = "";
            $.each(me.storearr, function(i, a) {
                str += (i == 0 ? "" : ",") + '{"q":"' + a.q + '","p":"' + a.p + '"}'
            });
            me.storestr = "[" + str + "]";
            bds.se.store.set("BDSUGSTORED", me.storestr)
        },
        mergeData: function(data) {
            var me = this;
            var arr = [];
            var index = 0;
            me.rsv_sug = 0;
            if (me.dataAladdin.length > 0) {
                for (var i = 0; i < me.dataAladdin.length; i++) {
                    arr.push(buildSug(me.dataAladdin[0].value, "ala", me.dataAladdin[0].otherData));
                    index++;
                    if (index >= 1) {
                        break
                    }
                }
            }
            if (me.dataStored.length > 0) {
                for (var i = 0; i < me.dataStored.length; i++) {
                    var duplicate = false;
                    for (var j = 0; j < arr.length; j++) {
                        if (me.dataStored[i].value == arr[j].value) {
                            duplicate = true
                        }
                    }
                    if (!duplicate) {
                        arr.push(buildSug(me.dataStored[i].value, "store", me.dataStored[i].pinyin));
                        index++
                    }
                    me.rsv_sug++;
                    if (index >= 2) {
                        break
                    }
                }
            }
            var arrNormal = [];
            if (data.s && data.s.length > 0) {
                for (var i = 0; i < data.s.length; i++) {
                    var duplicate = false;
                    for (var j = 0; j < arr.length; j++) {
                        if (data.s[i] == arr[j].value) {
                            duplicate = true
                        }
                    }
                    if (!duplicate) {
                        arrNormal.push(buildSug(data.s[i]))
                    }
                }
                arr = arr.concat(arrNormal)
            }
            return arr;
            function buildSug(v, type, otherData) {
                var obj = {};
                obj.value = v;
                var escapedV = setBold(v, me.queryValue);
                switch (type) {
                case"ala":
                    obj.html = "<h3>" + escapedV + "</h3><p>" + otherData.word + "</p>";
                    obj.type = "ala";
                    obj.alaid = otherData.id;
                    break;
                case"store":
                    obj.html = "<span>" + escapedV + '</span><u class="bdsug-store-del" title="如您不需要此搜索历史提示，&#13;可在右上角搜索设置中关闭">删除</u>';
                    obj.type = "store";
                    obj.pinyin = otherData;
                    break;
                default:
                    obj.html = escapedV;
                    break
                }
                if (!me.withoutZhixin && me.zhixinsug.length > 0) {
                    for (var z = 0; z < me.zhixinsug.length; z++) {
                        if (obj.value == me.zhixinsug[z].value && obj.type != "ala") {
                            obj.zxtype = me.zhixinsug[z].type;
                            obj.zxurl = me.zhixinsug[z].url;
                            var arrowhtml = '<i class="bdsug-arrow"></i>';
                            if (obj.html.split(arrowhtml).length <= 1) {
                                obj.html += arrowhtml
                            }
                        }
                    }
                }
                return obj;
                function setBold(str, q) {
                    if (q && str) {
                        str = $.trim(str);
                        q = $.trim(q);
                        if (str.indexOf(q) == 0 && str !== q) {
                            str = boldText(str, q)
                        } else {
                            str = escapeHTML(str)
                        }
                    }
                    return str
                }
                function boldText(text, key) {
                    text = escapeHTML(text);
                    key = escapeHTML(key);
                    var str_begin = key;
                    var len = key.length;
                    var str_end = "<b>" + text.substring(len) + "</b>";
                    return (str_begin + str_end)
                }
                function escapeHTML(str) {
                    str = str.replace(/&/g, "&amp;");
                    str = str.replace(/</g, "&lt;");
                    str = str.replace(/>/g, "&gt;");
                    return str
                }
            }
        },
        render: function(dataArray) {
            var me = this;
            me.is_selecting = false;
            if (!me.sugcontainer) {
                me.sugcontainer = document.createElement("DIV");
                me.sugcontainer.className = "bdsug";
                $(me.sugcontainer).delegate("li", "mouseenter", function() {
                    var v = $(this).data("key");
                    me.blurSug($(me.sugcontainer).find("li"));
                    me.focusSug(this, v);
                    me.index = $(me.sugcontainer).find("li").index($(this))
                }).delegate("li", "mouseleave", function() {
                    $(this).removeClass("bdsug-s")
                }).delegate("li", "click", function() {
                    var v = $(this).data("key");
                    me.sugValue = me.showValue = me.value = me.ipt.value = v;
                    me.index = $(me.sugcontainer).find("li").index($(this));
                    me.hideSug();
                    me.addStat("oq", me.inputValue);
                    me.addStat("sug", v);
                    me.addStat("rsv_sug2", 1);
                    me.formSubmit()
                });
                $(me.sugcontainer).click(function(e) {
                    e.stopPropagation()
                })
            }
            if (dataArray.length > 0) {
                var suglist = document.createElement("UL");
                var sclickUrl = bds && bds.util && bds.util.domain ? bds.util.domain.get("http://sclick.baidu.com"): "http://sclick.baidu.com";
                for (var i = 0; i < dataArray.length; i++) {
                    var sugli = document.createElement("LI");
                    sugli.innerHTML = dataArray[i].html;
                    if (dataArray[i].zxtype && dataArray[i].zxurl) {
                        var v = dataArray[i].value;
                        var t = dataArray[i].zxtype;
                        var url = dataArray[i].zxurl;
                        var hasData = (me.zhixindata[v] && me.zhixindata[v].responseJSON && me.zhixindata[v].responseJSON.status == 0 && me.zhixindata[v].responseJSON.data && me.zhixindata[v].responseJSON.data.length > 0);
                        if (!hasData) {
                            me.zhixindata[v] = $.ajax({
                                dataType: "jsonp",
                                async: true,
                                url: url,
                                jsonp: "cb"
                            })
                        }
                        $(sugli).addClass("bdsug-zx").on("focused", function() {
                            var v = $(this).data("key");
                            var hasData = (me.zhixindata[v] && me.zhixindata[v].responseJSON && me.zhixindata[v].responseJSON.status == 0 && me.zhixindata[v].responseJSON.data && me.zhixindata[v].responseJSON.data.length > 0);
                            var sugzxbox = $(me.sugcontainer).find(".bdsug-box")[0];
                            if (hasData && sugzxbox) {
                                if (bds.se.sugzx && bds.se.sugzx[t]) {
                                    var $sugzxbox = bds.se.sugzx[t](me.zhixindata[v].responseJSON.data, v, me.outInterface(), me);
                                    if ($sugzxbox instanceof jQuery) {
                                        $(me.sugcontainer).addClass("bdsug-showzx");
                                        $(sugzxbox).empty().append($sugzxbox);
                                        if ($(me.sugcontainer).height() < $(me.sugcontainer).find(".bdsug-box").height()) {
                                            $(me.sugcontainer).css({
                                                height: $(me.sugcontainer).find(".bdsug-box").height()
                                            })
                                        }
                                    }
                                }
                            }
                        }).on("blured", function() {
                            $(me.sugcontainer).removeClass("bdsug-showzx");
                            $(me.sugcontainer).css({
                                height: "auto"
                            })
                        })
                    }
                    me.setSug(sugli, dataArray[i].value, dataArray[i].type);
                    suglist.appendChild(sugli)
                }
                if (me.form) {
                    $(me.sugcontainer).insertBefore(me.form.firstChild)
                } else {
                    $(me.ipt).after(me.sugcontainer)
                }
                me.sugcontainer.innerHTML = "";
                me.sugcontainer.appendChild(suglist);
                $(me.sugcontainer).removeClass("bdsug-showzx");
                $(me.sugcontainer).css({
                    height: "auto"
                });
                var suglis = $(me.sugcontainer).find("ul li");
                if (!me.withoutZhixin) {
                    $(me.sugcontainer).addClass("bdsug-showarrow");
                    $(me.sugcontainer).append($('<div class="bdsug-box"></div>'));
                    if (!$(me.form).find(".bdsug-tmp").length) {
                        $(me.form).append($('<div class="bdsug-tmp"></div>'))
                    }
                    $(me.sugcontainer).find(".bdsug-box").on("mouseenter", function() {
                        $(me.sugcontainer).addClass("bdsug-showzx");
                        $(suglis[me.index]).addClass("bdsug-s")
                    }).on("mouseleave", function() {
                        $(me.sugcontainer).removeClass("bdsug-showzx");
                        $(suglis[me.index]).removeClass("bdsug-s");
                        $(me.sugcontainer).css({
                            height: "auto"
                        })
                    }).on("click", function(e) {
                        e.stopPropagation()
                    })
                }
                $(me.form).find(".bdsug-tmp").empty();
                var $lishort = $("<li>").appendTo($("<div>").addClass("bdsug-showzx").appendTo($(me.form).find(".bdsug-tmp")));
                var $div = $("<div>").css({
                    position: "absolute",
                    display: "inline-block",
                    top: "-10000px",
                    left: "-10000px"
                }).appendTo($(me.form).find(".bdsug-tmp"));
                $.each(suglis, function(i, li) {
                    $div.html($(li).html());
                    var cut = 28;
                    if ($div.width() > ($lishort.width() - $(me.sugcontainer).find(".bdsug-arrow").width() - cut)) {
                        $(li).addClass("bdsug-overflow")
                    }
                });
                $.each(suglis, function(i, li) {
                    if (dataArray[i] && dataArray[i].type == "store") {
                        var index = i;
                        $(li).find("u").click(function(e) {
                            e.stopPropagation();
                            $(li).remove();
                            var suginstore = false;
                            $.each(me.storearr, function(dex, arr) {
                                if (me.dataArray[index].pinyin == decodeURIComponent(me.storearr[dex].p)) {
                                    suginstore = dex
                                }
                            });
                            if (suginstore !== false) {
                                me.storearr.splice(suginstore, 1);
                                me.setStore()
                            }
                            var img = window["BD_PS_C" + (new Date()).getTime()] = new Image();
                            img.src = sclickUrl + "/w.gif?q=" + encodeURIComponent(dataArray[index].value) + "&fm=beha&rsv_sug=del&rsv_sid=" + bds.comm.sid + "&t=" + new Date().getTime() + "&path=http://www.baidu.com"
                        })
                    }
                });
                $(me.ipt).trigger("render", [me]);
                me.renderCallback();
                me.showSug();
                $(me.ipt).trigger("showSug", [me])
            } else {
                me.renderCallback();
                me.hideSug()
            }
        },
        setAutocomplete: function(c) {
            var me = this;
            $(me.ipt).attr("autocomplete", c)
        },
        setSug: function(li, v, type) {
            var me = this;
            $(li).attr("data-key", v);
            if (type) {
                $(li).addClass("bdsug-" + type)
            }
        },
        clickIpt: function() {
            var me = this;
            me.request(me.ipt.value, "2")
        },
        showSug: function() {
            var me = this;
            me.index =- 1;
            if (me.ipt.value == me.reqValue) {
                $(me.sugcontainer).show()
            }
            me.visible = true;
            me.rsv_sug1++;
            me.addStat("rsv_sug1", me.rsv_sug1)
        },
        hideSug: function() {
            var me = this;
            me.is_selecting = false;
            $(me.ipt).trigger("hide", [me]);
            $(me.sugcontainer).hide();
            me.visible = false;
            if (me.jqXhr) {
                me.jqXhr.abort()
            }
            me.jqXhr = null
        },
        selectSug: function(i) {
            var me = this;
            var lis = $(me.sugcontainer).find("li");
            me.blurSug(lis);
            if (i!=-1) {
                me.is_selecting = true;
                var v = $($(lis)[i]).data("key");
                me.focusSug(lis[i], v);
                me.sugValue = me.showValue = me.value = me.ipt.value = v;
                me.addStat("sug", v);
                me.addStat("oq", me.inputValue);
                me.addStat("rsv_n", 1)
            } else {
                me.is_selecting = false;
                me.selectCallback(me.inputValue);
                me.showValue = me.value = me.ipt.value = me.inputValue;
                me.sugValue = "";
                me.removeStat("sug");
                me.removeStat("oq");
                me.removeStat("rsv_n")
            }
            $(me.ipt).trigger("selectSug", [me, i, v])
        },
        blurSug: function(lis) {
            var me = this;
            $(lis).removeClass("bdsug-s");
            $(lis).trigger("blured")
        },
        focusSug: function(li, v) {
            var me = this;
            $(li).addClass("bdsug-s");
            $(li).trigger("focused");
            me.selectCallback(v)
        },
        pressUp: function(n) {
            var me = this;
            var n = $(me.sugcontainer).find("li").length;
            me.index--;
            if (me.index<-1) {
                me.index = n - 1
            }
            me.selectSug(me.index)
        },
        pressDown: function(n) {
            var me = this;
            var n = $(me.sugcontainer).find("li").length;
            me.index++;
            if (me.index >= n) {
                me.index =- 1
            }
            me.selectSug(me.index)
        },
        addStat: function(name, value) {
            var me = this;
            if (!me.stat) {
                me.stat = {}
            }
            me.stat[name] = value;
            if (me.withoutStat) {
                return 
            }
            var ipthidden = $(me.form).find('input[type="hidden"][name="' + name + '"]');
            if (ipthidden.length) {
                $(ipthidden).val(value)
            } else {
                $(me.form).append('<input type="hidden" name="' + name + '" value="' + value + '"/>')
            }
        },
        removeStat: function(name) {
            var me = this;
            if (me.withoutStat) {
                return 
            }
            $(me.form).find('input[type="hidden"][name="' + name + '"]').remove();
            if (!me.stat) {
                try {
                    delete me.stat[name]
                } catch (e) {}
            }
        },
        clearStat: function() {
            var me = this;
            me.removeStat("rsp");
            me.removeStat("rsv_n");
            me.removeStat("rsv_sug");
            me.removeStat("rsv_sug1");
            me.removeStat("rsv_sug2");
            me.removeStat("rsv_sug3");
            me.removeStat("rsv_sug4");
            me.removeStat("rsv_sug5");
            me.removeStat("rsv_sug6");
            me.stat = {}
        },
        formSubmit: function(ev) {
            var me = this, ret = true;
            if (me.index!=-1) {
                me.addStat("f", 3);
                me.addStat("rsp", me.index);
                if (me.dataArray[me.index] && me.dataArray[me.index].alaid) {
                    me.addStat("rsv_sug5", me.dataArray[me.index].alaid)
                } else {
                    me.removeStat("rsv_sug5")
                }
            }
            if (me.sugValue == me.value) {
                me.removeStat("rsv_n");
                me.removeStat("sug")
            }
            if (me.inputT != 0) {
                me.addStat("inputT", new Date().getTime() - me.inputT);
                me.inputT = 0
            }
            me.rsv_sug && me.addStat("rsv_sug", me.rsv_sug);
            if ($($(me.sugcontainer).find("li")[me.index]).hasClass("bdsug-zx")) {
                if ($(me.sugcontainer).find(".bdsug-box").attr("bdsug-click") == "all") {
                    me.addStat("rsv_sugtp", 1)
                } else {
                    me.addStat("rsv_sugtp", 0)
                }
            } else {
                me.removeStat("rsv_sugtp")
            }
            try {
                (function() {
                    var _t = new Date().getTime();
                    document.cookie = "WWW_ST=" + _t + ";expires=" + new Date(_t + 10000).toGMTString()
                })()
            } catch (e) {}
            if ($.isFunction(me.submission)) {
                ret = me.submission.call(me.form, ev);
                me.hideSug();
                me.inputValue = me.value;
                me.clearStat()
            }
            if (ret) {
                me.form.submit()
            }
        },
        init: function() {
            var me = this;
            me.setAutocomplete("off");
            me.updateInitData();
            if (!bds.se.sugdnscached) {
                $.ajax({
                    dataType: "jsonp",
                    async: true,
                    scriptCharset: "gbk",
                    url: me.buildUrl("", "", true),
                    jsonp: "cb",
                    timeout: 5000,
                    success: function(data) {}
                });
                bds.se.sugdnscached = 1
            }
            me.startCircle();
            me.clearStat();
            $(me.ipt).on("click", function(e) {
                e.stopPropagation();
                !me.withoutMode && me.clickIpt()
            }).on("focus", function() {
                me.startCircle()
            }).on("keydown", function(e) {
                e = e || window.event;
                if (me.inputT == 0) {
                    me.inputT = new Date().getTime()
                }
                if (e.keyCode == 9 || e.keyCode == 27) {
                    me.hideSug()
                }
                if (e.keyCode == 13) {
                    me.addStat("rsv_sug2", 0)
                }
                if (e.keyCode == 86 && e.ctrlKey) {
                    me.addStat("rsv_n", 2)
                }
                if (me.sugcontainer && me.sugcontainer.style.display != "none") {
                    if (e.keyCode == 38) {
                        e.preventDefault();
                        me.pressUp()
                    }
                    if (e.keyCode == 40) {
                        e.preventDefault();
                        me.pressDown()
                    }
                } else {
                    if (e.keyCode == 38 || e.keyCode == 40) {
                        e.preventDefault();
                        me.request(me.value)
                    }
                }
            });
            me.form && $(me.form).submit(function(e) {
                me.formSubmit(e);
                return false
            })
        },
        outInterface: function() {
            var me = this;
            return {
                setVisibleSug: (function() {
                    var icon = $("<i class='c-icon c-icon-bear-round'></i>");
                    return function(value) {
                        var i;
                        icon.remove();
                        for (i = 0; i < me.dataArray.length; i++) {
                            if (me.dataArray[i].value == value) {
                                icon.appendTo($(me.sugcontainer).find("li").eq(i));
                                break
                            }
                        }
                    }
                })(),
                getStat: function(name) {
                    if (me.stat) {
                        return me.stat[name]
                    }
                },
                clearStat: function() {
                    return me.clearStat()
                },
                setKey: function(wd) {
                    me.ipt.value = me.value = me.inputValue = me.showValue = wd
                },
                visible: function() {
                    return me.visible
                },
                is_selecting: function() {
                    return me.visible && me.is_selecting
                },
                data: function() {
                    return me.dataArray
                },
                show: function() {
                    return me.showSug()
                },
                hide: function() {
                    return me.hideSug()
                },
                start: function() {
                    return me.startCircle()
                },
                stop: function() {
                    return me.stopCircle()
                },
                setMaxNum: function(maxNum) {
                    return me.maxNum = maxNum
                },
                check: function() {
                    return me.check()
                },
                formSubmit: function() {
                    return me.formSubmit()
                },
                updateInitData: function() {
                    return me.updateInitData()
                },
                on: function() {
                    $(me.ipt).on.apply($(me.ipt), arguments)
                },
                height: function() {
                    return $(me.sugcontainer).height()
                },
                off: function() {
                    $(me.ipt).off.apply($(me.ipt), arguments)
                }
            }
        }
    }
})();
