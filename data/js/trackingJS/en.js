function _ge(n) {
    return _d.getElementById(n)
}
function sj_wf(n) {
    var t = arguments;
    return function() {
        n.apply(null, [].slice.apply(t).slice(1))
    }
}
function sj_ce(n, t, i) {
    var r = _d.createElement(n);
    return t && (r.id = t), i && (r.className = i), r
}
function sj_be(n, t, i, r) {
    if ((n == _w || n == _d.body) && t == "load" && "undefined" != typeof _d.readyState && "complete" === _d.readyState) {
        i();
        return 
    }
    n.addEventListener ? n.addEventListener(t, i, r) : n.attachEvent ? n.attachEvent("on" + t, i) : n["on" + t] = i
}
function sj_ue(n, t, i, r) {
    n.removeEventListener ? n.removeEventListener(t, i, r) : n.detachEvent ? n.detachEvent("on" + t, i) : n["on" + t] = null
}
function sj_ev(n) {
    return sb_ie ? event : n
}
function sj_et(n) {
    return sb_ie ? event.srcElement : n.target
}
function sj_mi(n) {
    return sb_ie ? event.fromElement : n.relatedTarget
}
function sj_mo(n) {
    return sb_ie ? event.toElement : n.relatedTarget
}
function sj_pd(n) {
    sb_ie ? event.returnValue=!1 : n.preventDefault()
}
function sj_sp(n) {
    sb_ie ? n.cancelBubble=!0 : n.stopPropagation()
}
function sj_we(n, t, i) {
    while (n && n != (i || sj_b)) {
        if (n == t)
            return !0;
        n = n.parentNode
    }
    return !1
}
function sj_ic(n) {
    var t = sj_ce("style");
    _d.getElementsByTagName("head")[0].appendChild(t), t.textContent !== undefined ? t.textContent = n : t.styleSheet.cssText = n
}
_w = window, _d = document, sb_de = _d.documentElement, sb_ie=!!_w.ActiveXObject, sb_i6 = sb_ie&&!_w.XMLHttpRequest, sb_st = _w.setTimeout, sb_ct = _w.clearTimeout, sb_gt = function() {
    return + new Date
}, sj_evt = new function() {
    function i(n) {
        return t[n] || (t[n] = [])
    }
    var t = {}, n = this;
    n.fire = function(n) {
        for (var r = i(n), u = r.e = arguments, t = 0; t < r.length; t++)
            r[t].d ? sb_st(sj_wf(r[t], u), r[t].d) : r[t](u)
    }, n.bind = function(n, t, r, u) {
        var f = i(n);
        t.d = u, f.push(t), r && f.e && t(f.e)
    }, n.unbind = function(n, i) {
        for (var u = 0, r = t[n]; r && u < r.length; u++)
            if (r[u] == i) {
                r.splice(u, 1);
                break
            }
    }
}, sj_cook = new function() {
    var n = this;
    n.get = function(n, t) {
        var i = _d.cookie.match(new RegExp("\\b" + n + "=[^;]+")), r;
        return t && i ? (r = i[0].match(new RegExp("\\b" + t + "=([^&]*)")), r ? r[1] : null) : i ? i[0] : null
    }, n.set = function(t, i, r, u, f, e) {
        var c, l = i + "=" + r, h = n.get(t), o, s, a;
        h ? (o = n.get(t, i), c = o ? h.replace(i + "=" + o, l) : h + "&" + l) : c = t + "=" + l, s = location.hostname.match(/([^.]+\.[^.]*)$/), a = e && e > 0 ? e * 6e4 : 63072e6, _d.cookie = c + (s ? ";domain=" + s[0] : "") + (u ? ";expires=" + new Date( + new Date + Math.min(a, 63072e6)).toGMTString() : "") + (f ? ";path=" + f : "")
    }
}, sj_b = _d.body, _w.Bing = _w.Bing || {}, Bing.AS = {}, Bing.AS.EventIds = {
    TextChanged: 1,
    SelectPrevItem: 2,
    SelectNextItem: 3,
    UpdateText: 4,
    NavigateSerp: 7,
    ReadyToProcessEvents: 8,
    SelectLeftItem: 9,
    SelectRightItem: 10,
    NavigateLink: 11,
    EnterKeyPressed: 12,
    EscKeyPressed: 13,
    CloseFlyout: 15,
    LastSharedEventId: 99,
    ConfigurationSet: 100,
    Initialized: 101,
    RequestSent: 102,
    ApiCallback: 103,
    ApiServiceReceived: 104,
    ValidResponseReceived: 105,
    SuggestionsRendered: 106,
    NoResults: 107,
    DropDownShown: 108,
    DropDownHidden: 109,
    BeforeNavigate: 110,
    UpdateTextAndRequery: 111
}, Bing.AS.Constants = {
    KeyCodeBackspace: 8,
    KeyCodeTab: 9,
    KeyCodeEscape: 27,
    KeyCodeSpace: 32,
    KeyCodeArrowLeft: 37,
    KeyCodeArrowUp: 38,
    KeyCodeArrowRight: 39,
    KeyCodeArrowDown: 40,
    KeyCodeDelete: 46,
    KeyCodeImeOpen: 229,
    HitHighlightStartToken: "<strong>",
    HitHighlightEndToken: "</strong>"
}, Bing.AS.AutoSuggest = function() {
    function l() {
        if (n.currentSuggestion && n.currentSuggestion.horizParentSug) {
            var t = n.currentSuggestion.horizParentSug;
            n.unHighlightSuggestionHandler(), n.currentSuggestion = t
        }
    }
    function o(t) {
        for (var i = n.currentSuggestion; i = i[t];)
            if (!i.hd)
                return i;
        return null
    }
    function f(t) {
        !t || t && t.supressTextOnSelect ? n.raiseEvent(Bing.AS.EventIds.UpdateText, n.currentQuery, !1) : n.raiseEvent(Bing.AS.EventIds.UpdateText, n.currentSuggestionList[t.ord], !1), n.selectedSuggestion = t
    }
    function y(t, i) {
        var r = n.selectedSuggestion;
        switch (t) {
        case"sc":
            return n.currentSuggestionList.length + "-" + d.length;
        case"pq":
            return i ? encodeURIComponent(n.currentQuery) : n.currentQuery;
        case"sp":
            return r ? r.ord + 1 : - 1;
        case"qs":
            return r ? r.src : "n";
        case"sk":
            return r && r.sk ? r.sk : ""
        }
    }
    function p(t, i) {
        var h, s, p, e, l, f, u, nt, w, a, c;
        _drawer.innerHTML = "", n.currentSuggestionList = [];
        var o = [], h, v = [], y = 0, tt = "", b = g[t[0].Type];
        for (r.appendHistory(i, t, b), s = 0; s < t.length; s++)
            h = t[s], f = h.Type, v[o.length] = f, p = h.Suggests, ut[f] || (o = o.concat(p)), n.raiseEvent(Bing.AS.EventIds.ApiServiceReceived, h);
        for (n.currentRenderIndex = 0, n.suggestionTypesRendered = {}, e = 0, w = 0, a = Math.min(o.length, k); e < a; e++) {
            if (u = o[e], !u)
                break;
            u.DisplayTxt = u.Txt, (f = v[e]) && n.HeadingBuilders[f] && (l = n.HeadingBuilders[f].call(n)) && n.addHeading(l, e == 0), n.suggestionTypesRendered[u.Type] == null && (n.suggestionTypesRendered[u.Type] = 0), c = rt(u, y), c && (n.suggestionTypesRendered[u.Type]++, y += c, n.currentRenderIndex++)
        }
        d = i, n.raiseEvent(Bing.AS.EventIds.SuggestionsRendered, i, _drawer.childNodes), sb_st(et, 1)
    }
    function et() {
        for (var u = _drawer.childNodes, i, t, n, r = 0; r < u.length; r++)
            if (i = u[r], i.overflowElems)
                for (t = 0; t < i.overflowElems.length; t++)
                    n = i.overflowElems[t], n.offsetLeft + n.offsetWidth > _drawer.offsetWidth && (n.style.display = "none")
    }
    function rt(t, i) {
        var e = t.Type, f = n.SuggestionGroupBuilders[e], u, r;
        return f ? f.call(n, t, i) : (u = n.SuggestionBuilders[t.Type] || n.buildSuggestion, r = sj_ce("li"), n.decorateSuggestion(r, t, i), r = u.call(n, t, r), !r)?!1 : (t.SelectTxt || (t.SelectTxt = t.DisplayTxt), n.appendSuggestion(r, t), !0)
    }
    function tt(t, i, r) {
        for (var f = [], u = 0, e, o = t.split(Bing.AS.Constants.HitHighlightStartToken), u = 0; u < o.length; u++)
            f = f.concat(o[u].split(Bing.AS.Constants.HitHighlightEndToken));
        for (u = 0; u < f.length; u++)
            f[u] && (e = u%2 == 0 ? n.buildSuggestionSegment(f[u], r ? "span" : 0, r) : n.buildSuggestionSegment(f[u], "strong", r), i.appendChild(e))
    }
    function a() {
        return t.hn ? "http://" + t.hn : ""
    }
    function u(n) {
        n = sj_ev(n), sj_sp(n)
    }
    var n = this, e;
    n.config = null, n.drawer = null, n.currentQuery = null, n.currentRenderIndex = 0, n.currentSuggestion, n.currentSuggestionList = [], n.selectedSuggestion, n.clickedSuggestion, n.suggestionTypesRendered, n.apiOptions = "", n.baseQuery = "", n.SuggestionClassName = "sa_sg";
    var v = "sa_hv", h = "sa_tm", ot = "sa_sc", i = ["pq", "sc", "sp", "qs", "sk"], t, nt, c, b = 0, k, g = {}, ut = {
        QR: 1
    }, w = {}, d = "", s = {}, r, ft = new RegExp(Bing.AS.Constants.HitHighlightStartToken, "gi"), it = new RegExp(Bing.AS.Constants.HitHighlightEndToken, "gi");
    for (n.init = function(i, u, f) {
        var o, e, s;
        if (n.config = t = i, n.drawer = _drawer = u, n.baseQuery = f, nt = i.globalId, k = i.m || 5, i.p0)
            for (o = i.p0.split(","), e = 0; e < o.length; e++)
                g[o[e]] = 1;
        c = [i.u, "&mkt=", i.mkt, "&type=cb&cb=", nt, ".apiCB"].join(""), n.apiOptions = "", i.ePN && n.addApiOption("p"), i.eLO && n.addApiOption("l"), (i.eDN || i.dnx) && n.addApiOption("d"), i.eAN && (s = i.as || "a", n.addApiOption(s)), (i.eEQ || i.eID) && (n.addApiOption("eq"), n.InstrumentationTypes.push("EQ")), r = new Bing.AS.History(i), r.init(), n.raiseEvent(Bing.AS.EventIds.Initialized, this)
    }, n.addApiOption = function(t) {
        n.apiOptions += t + "+"
    }, n.apiCB = function(t) {
        var u, e;
        n.raiseEvent(Bing.AS.EventIds.ApiCallback);
        var r = "", i = [], o, f = t.AS;
        try {
            if (r = f.Query, u = "_" + r, e = w[u], e && _d.getElementsByTagName("head")[0].removeChild(e), w[u] = null, o = f.FullResults == 1, i = f.Results, i && o && (s[r] = i), n.currentQuery && r != n.currentQuery)
                return;
            if (!i) {
                n.raiseEvent(Bing.AS.EventIds.NoResults), n.currentSuggestionList = [];
                return 
            }
        } catch (h) {
            return 
        }
        n.raiseEvent(Bing.AS.EventIds.ValidResponseReceived, r, i), p(i, r)
    }, n.navigate = function(t) {
        var i = n.selectedSuggestion || n.currentSuggestion, r;
        i ? (r = i.query, i.url ? n.raiseEvent(Bing.AS.EventIds.NavigateLink, i.url + (i.addInst ? n.getAllSuggestionInstUrl() : ""), i.redir, r) : n.raiseEvent(Bing.AS.EventIds.NavigateSerp, r, t)) : n.raiseEvent(Bing.AS.EventIds.NavigateSerp, n.currentQuery, t)
    }, n.selectPrevious = function() {
        l();
        var t = _drawer.lastChild;
        if (n.currentSuggestion&&!(t = o("previousSibling"))) {
            f(), n.unHighlightSuggestionHandler(), n.raiseEvent(Bing.AS.EventIds.NoItemSelected);
            return 
        }
        f(t), n.highlightSuggestion(t)
    }, n.selectNext = function() {
        l();
        var t;
        if (!(n.currentSuggestion && (t = o("nextSibling")))) {
            if (n.currentSuggestion) {
                f(), n.unHighlightSuggestionHandler(), n.raiseEvent(Bing.AS.EventIds.NoItemSelected);
                return 
            }
            t = n.currentSuggestion = _drawer.firstChild, t.hd && (t = o("nextSibling"))
        }
        f(t), n.highlightSuggestion(t)
    }, n.getAllSuggestionInstUrl = function() {
        for (var t = [], n = 0; n < i.length; n++)
            t = t.concat("&", i[n], "=", y(i[n], !0));
        return t.join("")
    }, n.getAllSuggestionInstObject = function() {
        for (var t = {}, n = 0; n < i.length; n++)
            t[i[n]] = y(i[n], !1);
        return t
    }, n.setCursorPosition = function(n) {
        b = n
    }, n.fetch = function(i) {
        var f;
        if (!t.ts && s[i]&&!t.eRS) {
            n.raiseEvent(Bing.AS.EventIds.ApiCallback), p(s[i], i);
            return 
        }
        var e = n.apiOptions, u = [c, "&q=", encodeURIComponent(i), "&cp=" + b], o = t.ts;
        t.h && n.currentQuery != "" && (e += "h+"), f = r.getApiParameters(n.currentQuery), f && (u.push(f), e += r.getApiOptions(n.currentQuery), o=!0), o && u.push("&qt=", sb_gt()), n.baseQuery != "" && u.push("&bq=", encodeURIComponent(n.baseQuery)), t.ds && u.push("&ds=", t.ds), t.cnt && t.cnt > 0 && u.push("&count=", t.cnt), Bing.AS.Utils.LoadScript(u.concat("&o=", e.slice(0, - 1)).join("")), n.raiseEvent(Bing.AS.EventIds.RequestSent, n.currentQuery)
    }, n.addHeading = function(n, t) {
        var i = sj_ce("li", "", "sa_hd" + (t ? "_first" : ""));
        i.hd=!0, i.innerHTML = n, this.drawer.appendChild(i)
    }, n.decorateSuggestion = function(t, i, r) {
        t.src = i.Type, t.ord = r, t.sk = i.Sk, sj_be(t, "mouseover", function(i) {
            u(i), n.highlightSuggestion(t)
        }), sj_be(t, "mouseout", function(t) {
            u(t), n.unHighlightSuggestionHandler()
        }), sj_be(t, "mousedown", function(i) {
            u(i);
            n.onMouseDownSuggestion(i, t)
        }), sj_be(t, "mouseup", function(i) {
            u(i);
            n.onMouseUpSuggestion(t)
        })
    }, n.appendSuggestion = function(t, i) {
        var r = i.SelectTxt.replace(ft, "").replace(it, "");
        t.query = r, _drawer.appendChild(t), n.currentSuggestionList.push(r)
    }, n.buildSuggestionTerm = function(n, t, i) {
        var u = h, r;
        n.isHist && (u += " " + h + "HS"), r = sj_ce(i || "div", "", u), n.Type == "ID" && (r.lang = "en"), t.appendChild(r), tt(n.DisplayTxt, r)
    }, n.buildSuggestionDescription = function(t, i, r) {
        t.Meta1 && i.appendChild(n.buildSuggestionSegment(t.Meta1, r || "div", "sa_ds"))
    }, n.buildSuggestionImage = function(n, t, i) {
        if (n.Img) {
            var f = sj_ce(i || "div", "", "sa_img"), r = sj_ce("img"), u = n.Img.alt;
            u && (r.title = u), r.src = a() + n.Img.src, r.width = n.Img.w, r.height = n.Img.h, f.appendChild(r), t.appendChild(f)
        }
    }, n.addSuggestionUrl = function(n, i) {
        n.Url && (i.url = a() + n.Url + (t.fc ? "&FORM=" + t.fc : ""), i.addInst=!0)
    }, n.buildSuggestionSegment = function(n, t, i) {
        var r = t ? sj_ce(t, "", i): _d.createTextNode(n);
        return t && Bing.AS.Utils.SetInnerText(r, n), r
    }, n.highlightSuggestion = function(t) {
        n.unHighlightSuggestionHandler(), t.className = t.className + " " + v, n.currentSuggestion = t
    }, n.unHighlightSuggestionHandler = function() {
        n.currentSuggestion && (n.currentSuggestion.className = n.currentSuggestion.className.replace(" " + v, ""), n.currentSuggestion = null)
    }, n.onMouseDownSuggestion = function(t, i) {
        t = sj_ev(t);
        var r = t.button;
        r == 2 ? (u(t), sj_pd(t)) : n.clickedSuggestion = i
    }, n.onMouseUpSuggestion = function(t) {
        n.clickedSuggestion && n.clickedSuggestion == t && (f(n.clickedSuggestion), n.navigate(), n.clickedSuggestion = null)
    }, e = 0; e < Bing.AS.AutoSuggest.ScopeExtensions.length; e++)
        Bing.AS.AutoSuggest.ScopeExtensions[e].init.apply(this)
}, Bing.AS.AutoSuggest.ScopeExtensions = [], Bing.AS.AutoSuggest.prototype.EventRegistry = [], Bing.AS.AutoSuggest.prototype.InstrumentationTypes = [], Bing.AS.AutoSuggest.prototype.HeadingBuilders = {}, Bing.AS.AutoSuggest.prototype.SuggestionBuilders = {}, Bing.AS.AutoSuggest.prototype.SuggestionGroupBuilders = {}, Bing.AS.AutoSuggest.prototype.buildSuggestion = function(n, t) {
    return t.className = this.SuggestionClassName + (this.drawer.childNodes.length == 0 ? " sa_sg_first " : " ") + this.SuggestionClassName + n.Type, this.buildSuggestionTerm(n, t), this.addSuggestionUrl(n, t), t.hc = n.HC, t
}, Bing.AS.AutoSuggest.prototype.setQuery = function(n) {
    var t = this;
    t.selectedSuggestion = null, t.currentSuggestion && (t.unHighlightSuggestionHandler(), t.currentSuggestion = null), t.currentQuery = n, t.fetch(n)
}, Bing.AS.AutoSuggest.prototype.HeadingBuilders.PN = function() {
    return this.currentQuery == "" && this.config.ePN ? sa_loc.H_PN : null
}, Bing.AS.AutoSuggest.prototype.HeadingBuilders.AS = function() {
    return this.config.eWH ? sa_loc.H_AS : null
}, Bing.AS.AutoSuggest.prototype.HeadingBuilders.EQ = function() {
    return this.config.eID ? sa_loc.H_ID : this.config.eEQ ? sa_loc.H_EQ : null
}, Bing.AS.EventRegistration = function() {
    this.events = [], this.add = function(n, t) {
        this.events[n] = t
    }
}, Bing.AS.AutoSuggest.prototype.raiseEvent = function(n) {
    for (var r = arguments.length ? [].slice.apply(arguments).slice(1) : [], i, t = 0; t < this.EventRegistry.length; t++)
        i = this.EventRegistry[t], i.events[n] && i.events[n].apply(this, r)
}, Bing.AS.History = function(n) {
    function f(n) {
        return !e ||!t.historyFromApi?!1 : i == null?!0 : n.indexOf(i) != 0
    }
    var t = this;
    t.config = n, t.historyFromApi=!0;
    var u = n.m, e = n.eHS, i = null, o = 0, r = "&sid=" + n.sid;
    n.maxHS != null && n.maxHS > 0 && (r += "&hs.count=" + n.maxHS), n.vid && (r += "&hs.ds=" + n.vid), t.getApiParameters = function(n) {
        return f(n) ? r : null
    }, t.getApiOptions = function(n) {
        return f(n) ? "hs+" : ""
    }, t.appendHistory = function(r, f, e) {
        var p = 0, o = 0, l = n.minHS, s = {}, h, y, w, c, a, v;
        for (historySuggestions = t.getHistory(r, f, u), s.Type = "HS", s.Suggests = historySuggestions, o = historySuggestions.length, h = 0; h < f.length; h++)
            for (y = f[h].Suggests, c = 0; c < y.length; c++) {
                if (f[h].Type != "HS")
                    for (a = 0; a < o; a++)
                        historySuggestions[a].Txt == y[c].Txt && f[h].Suggests.splice(c, 1);
                        p++
            }
        if (l = Math.max(l, u - (p - o)), o > l && (s.Suggests.splice(l, o - l), o = s.Suggests.length), o) {
            for (v = 0; v < o; v++)
                s.Suggests[v].isHist=!0;
            f.splice(e ? 1 : 0, t.historyFromApi ? 1 : 0, s)
        } else 
            i = r
    }
}, Bing.AS.History.prototype.init = function() {}, Bing.AS.History.prototype.getHistory = function(n, t) {
    for (var i = 0; i < t.length; i++)
        if (t[i].Type == "HS")
            return t[i].Suggests;
    return []
}, Bing.AS.AutoSuggest.ScopeExtensions.push(new function() {
    function t(n, t) {
        var i = null;
        return this.config.anc || (t.className = this.SuggestionClassName + " sa_2RowImg " + this.SuggestionClassName + n.Type, this.buildSuggestionImage(n, t), this.buildSuggestionTerm(n, t), this.buildSuggestionDescription(n, t), this.addSuggestionUrl(n, t), t.supressTextOnSelect=!0, i = t), i
    }
    var n = ["DT", "IG", "MV", "SA", "SI", "SN", "SP", "SR", "TC", "TR", "WE", "WH", "WM", "WU", "ID"];
    this.init = function() {
        for (var r = 0, i; r < n.length; r++)
            i = n[r], this.SuggestionBuilders[i] = t, this.InstrumentationTypes.push(i)
    }
}), Bing.AS.Utils = new function() {
    var t = new RegExp(Bing.AS.Constants.HitHighlightStartToken, "gi"), n = new RegExp(Bing.AS.Constants.HitHighlightEndToken, "gi");
    this.SetInnerText = function(n, t) {
        n[sj_b.textContent ? "textContent": "innerText"] = t
    }, this.GetComputedStyle = function(n) {
        return _w.getComputedStyle ? _w.getComputedStyle(n, null) : n.currentStyle
    }, this.GetQueryStringParam = function(n, t) {
        var r = new RegExp("[?&]{1}" + t + "=([^&]+)"), i = n.match(r);
        return i ? RegExp.$1 : null
    }, this.LoadScript = function(n) {
        var t = sj_ce("script");
        _d.getElementsByTagName("head")[0].appendChild(t), t.src = n
    }, this.SetSelectionRange = function(n, t, i) {
        if (n.setSelectionRange)
            n.setSelectionRange(t, i);
        else {
            var r = n.createTextRange();
            r.moveEnd("sentence", - 1e5), r.moveStart("character", t), r.moveEnd("character", i), r.select()
        }
    }
}, Bing.AS.SearchForm = function(n) {
    function p(n, i) {
        var s, e, o = t.autosuggest.getAllSuggestionInstObject(), r, f, u;
        for (r in o)
            s = _ge("sa_" + r), e = o[r], s ? s.value = e : t.addHiddenField(r, e);
        t.raiseEvent(Bing.AS.EventIds.BeforeNavigate), ft ? ft(t.searchBox, o) : (i || sj_evt.fire("onSearch", t.searchForm), t.fireEvt&&!i && (sj_ue(t.searchForm, "submit", w), t.fireEvt(t.searchForm, "submit"), sj_be(t.searchForm, "submit", w)), f = t.searchForm.getAttributeNode("onsubmit"), f && (u = f.value.match(/si_T\('(.*)'\)/), u && u[1] && si_T(u[1])), t.searchForm.submit())
    }
    function dt(n, t, r) {
        sb_st(i, 5), t ? (sj_cook.set("SCRHDN", "ASD", "1"), sj_cook.set("SCRHDN", "DURL", n), p(r)) : _w.location = n
    }
    function w(n) {
        sb_st(i, 5), k(n), _autosuggest.navigate(!0)
    }
    function ii(n) {
        t.cancelKeyEvent(n) || (n.keyCode == u.KeyCodeTab ? i() : b(n))
    }
    function b(n) {
        var f = n.keyCode;
        switch (f) {
        case u.KeyCodeEscape:
            i(), k(n);
            break;
        case u.KeyCodeArrowUp:
            r && (_autosuggest.selectPrevious(), t.raiseEvent(Bing.AS.EventIds.SelectPrevItem));
            break;
        case u.KeyCodeArrowDown:
            r ? (_autosuggest.selectNext(), t.raiseEvent(Bing.AS.EventIds.SelectNextItem)) : nt()
        }
    }
    function ri(n) {
        var i = n[1], r = i.keyCode;
        s || t.cancelKeyEvent(i) || (r == u.KeyCodeArrowUp || r == u.KeyCodeArrowDown || r == u.KeyCodeEscape) && t.handleUpDownEscOnKeyUp(i) && b(i)
    }
    function ti() {
        r || nt()
    }
    function nt() {
        (h && sb_ct(h), s) || (t.lastQuery = null, y())
    }
    function k(n) {
        sj_sp(n), sj_pd(n)
    }
    function gt() {
        e.style.width = t.getDrawerWidth(), f.style.left = t.getDrawerLeft(), r || (f.style.display = "block", e.style.height = "auto", r=!0, t.raiseEvent(Bing.AS.EventIds.DropDownShown))
    }
    function i() {
        r && (f.style.display = "none", bt = r=!1, t.lastQuery = o(), t.raiseEvent(Bing.AS.EventIds.DropDownHidden))
    }
    function ni() {
        h = sb_st(i, 50)
    }
    function st(n) {
        l=!1, t.searchBox.value = n
    }
    function et(n) {
        t.searchBox.value = n;
        var r = t.searchBox, i = n.length;
        r.focus(), Bing.AS.Utils.SetSelectionRange(r, i, i)
    }
    function g() {
        if (!s) {
            l && y(), l || (t.lastQuery = o(), l=!0);
            var n = it;
            rt && (n*=1<<Math.max(0, v - pt + 1) / 2)
        }
        sb_st(g, n)
    }
    function y() {
        var n = o();
        t.lastQuery != n && n.length >= ot && (h && sb_ct(h), _autosuggest.setCursorPosition(ct()), _autosuggest.setQuery(n), t.lastQuery = n, v++)
    }
    function lt() {
        v--
    }
    function ct() {
        var i = 0, r = _d.selection, u = t.searchBox.selectionStart, n;
        return u != null ? i = u : r && t.searchBox.createTextRange && (n = t.searchBox.createTextRange(), n.setEndPoint("EndToStart", r.createRange()), i = n.text.length), i
    }
    function o() {
        return t.searchBox.value.replace(wt, "").replace(yt, " ").toLowerCase()
    }
    function ht() {
        function t(t, i) {
            var f = n[t], u;
            f && e < kt && (u = sj_ce("a"), u.innerHTML = sa_loc[i], r[e++] = vt(u, f))
        }
        var u = sj_ce("div", "", "sa_om"), f = sj_ce("ul"), r = [], e = 0, i;
        for (u.appendChild(f), t("lads", "L_ADS"), t("lmh", "L_MH"), t("lp", "L_P"), t("lh", "L_H"), i = r.length; i > 0; i--)
            f.appendChild(r[i - 1]);
        return u
    }
    function vt(n, t, i, r, u) {
        var f = sj_ce("li", "", r);
        return n.handler = i, n.newWin = u, t && (n.href = t), u && (n.target = "_blank"), n.onclick = at, f.appendChild(n), f
    }
    function at() {
        var n=!this.newWin;
        this.handler && (n = n && this.handler()), n && i()
    }
    var t = this, c;
    t.raiseEvent(Bing.AS.EventIds.ConfigurationSet, n), t.config = n, t.searchForm = null, t.searchBox = null, t.autosuggest = null, t.optionsMenu = null, t.lastQuery = null, t.DemoQuery = "asdemoq", t.DemoPath = "asdemopath";
    var tt = "SRCHHPGUSR", d = "AS", pt = 4, kt = 2, a, f, e, it = 0, rt, v = 0, ot = n.eHS || n.ePN ? 0: 1, h, s, bt, r, l=!0, u = Bing.AS.Constants, wt = new RegExp("^\\s*", "g"), yt = new RegExp("\\s+", "g"), ft = typeof n.cb == "function" ? n.cb : null, ut;
    for (t.init = function(r) {
        var h, l, v, u, c;
        (t.config.globalId = r, s = t.isDisabled()) || (a = n.i, t.searchForm = _ge(n.f), t.searchBox = _ge(a), t.searchForm.form && (n.fc = t.searchForm.form.value), sj_cook.set("SCRHDN", "ASD", "0"), sj_cook.set("SCRHDN", "DURL", "#"), h = _ge(n.c || "sw_as"), f = sj_ce("div", "", "sa_as"), h.appendChild(f), e = sj_ce("div", "", "sw_b3"), f.appendChild(e), l = sj_ce("ul", "", "sa_drw"), e.appendChild(l), t.optionsMenu = ht(), e.appendChild(t.optionsMenu), h.style.display = "block", t.registerUIElement(h), t.registerUIElement(t.searchBox), v = _ge("sb_form_go"), v && t.registerUIElement(v), it = n.d, rt = n.t, t.lastQuery = o(), t.autosuggest = _autosuggest = new Bing.AS.AutoSuggest, _autosuggest.init(n, l, t.lastQuery), u = new Bing.AS.EventRegistration, u.add(Bing.AS.EventIds.ApiCallback, lt), u.add(Bing.AS.EventIds.NoResults, ni), u.add(Bing.AS.EventIds.SuggestionsRendered, gt), u.add(Bing.AS.EventIds.UpdateText, st), u.add(Bing.AS.EventIds.UpdateTextAndRequery, et), u.add(Bing.AS.EventIds.NavigateSerp, p), u.add(Bing.AS.EventIds.NavigateLink, dt), t.autosuggest.EventRegistry.push(u), sj_be(_w, "blur", i), sj_be(sj_b, "click", i), sj_be(t.searchForm, "submit", w), sj_be(t.searchBox, "keydown", ii), sj_evt.bind("onASKeyup" + a, ri, !0), sj_evt.bind("onASClick" + a, ti, !0), g(), t.raiseEvent(Bing.AS.EventIds.Initialized, t), c = _w.location.search, n.demo && (Bing.AS.Utils.GetQueryStringParam(c, t.DemoQuery) || Bing.AS.Utils.GetQueryStringParam(c, t.DemoPath)) && (Bing.AS.Utils.LoadScript("/fd/sa/" + _G.Ver + "/AutoSuggest/AutoSuggestDemo.js"), sj_evt.fire("onASDemo", t)))
    }, t.apiCB = function(n) {
        t.autosuggest.apiCB(n)
    }, t.isDisabled = function() {
        return sj_cook.get(tt, d) == 0
    }, t.enable = function(n, i) {
        n && (t.lastQuery = o()), s=!n, !i && sj_cook.set(tt, d, n ? 1 : 0, 1, "/")
    }, t.sid = function(t) {
        _autosuggest.config.sid = n.sid = t
    }, t.registerUIElement = function(n) {
        sj_be(n, "click", function(n) {
            sj_sp(n)
        })
    }, t.addHiddenField = function(n, i) {
        var r = sj_ce("input");
        return r.id = "sa_" + n, r.name = n, r.value = i, r.type = "hidden", t.searchForm.appendChild(r), r
    }, t.drawerIsOpen = function() {
        return r
    }, t.getDefaultDrawerWidth = function() {
        return t.searchBox.parentNode.offsetWidth - 2 + "px"
    }, c = 0; c < Bing.AS.SearchForm.ScopeExtensions.length; c++)
        Bing.AS.SearchForm.ScopeExtensions[c].init.apply(this)
}, Bing.AS.SearchForm.ScopeExtensions = [], Bing.AS.SearchForm.prototype.EventRegistry = [], Bing.AS.SearchForm.prototype.cancelKeyEvent = function() {
    return !1
}, Bing.AS.SearchForm.prototype.handleUpDownEscOnKeyUp = function() {
    return !1
}, Bing.AS.SearchForm.prototype.getDrawerLeft = function() {
    return 0
}, Bing.AS.SearchForm.prototype.getDrawerWidth = function() {
    return this.getDefaultDrawerWidth()
}, Bing.AS.SearchForm.prototype.raiseEvent = Bing.AS.AutoSuggest.prototype.raiseEvent, Bing.AS.SearchForm.prototype.fireEvt = function(n, t, i) {
    var r, u;
    try {
        n.dispatchEvent ? (u = "HTMLEvents", t != "submit" ? (u = "MouseEvents", r = _d.createEvent(u), r.initEvent(t, !0, !0, _w, 0, 0, 0, 0, 0, !1, !1, !1, !1, 0, i)) : (r = _d.createEvent(u), r.initEvent(t, !0, !0)), i && (r.sourceElement = i), n.dispatchEvent(r)) : i ? (r = _d.createEventObject(), r.sourceElement = i, n.fireEvent("on" + t, r)) : n.fireEvent("on" + t)
    } catch (f) {}
}, function() {
    function t(n) {
        n.m = 8, n.d = 100, n.cHS = 3, n.ml = 45, n.mh = 256, n.t = 1;
        var t = _ge(n.i);
        sj_be(t, "keyup", function(t) {
            var r = {}, i;
            for (i in t)
                r[i] = t[i];
            sj_evt.fire("onASKeyup" + n.i, r)
        }), sj_be(t, "click", sj_wf(sj_evt.fire, "onASClick" + n.i))
    }
    var n = new Bing.AS.EventRegistration;
    n.add(Bing.AS.EventIds.ConfigurationSet, t), Bing.AS.SearchForm.prototype.EventRegistry.push(n)
}(), sa_autosuggest = Bing.AS.SearchForm, sj_ic(".sa_as{display:none;position:absolute;top:-1px;left:0}.sa_as ul{list-style:none}.sw_b3{border:1px solid #ccc;background:#fff;overflow:visible}.sa_drw{padding:0;background:#fff;margin:0}.sa_drw li{display:block;white-space:nowrap}.sa_sg,.sa_hd,.sa_hd_first{font-size:120%;padding:.2em .55em .21em .55em;color:#777;cursor:pointer;clear:left}.sa_sg_first{padding-top:.5em}.sa_hd,.sa_hd_first{padding:.6em 0 .1em .52em;margin:.25em .2em 0 .2em;border-top:1px solid #ededed;text-transform:uppercase;font-size:95%;color:#555;cursor:default}.sa_hd_first{margin-top:0;padding-top:.6em;border:0}.sa_2RowImg{background:#f8f8f8}.sa_2RowImg{border-bottom:1px solid #ccc;padding:.4em .55em}.sa_ds{color:#737373;margin-top:.2em;font-size:90%}.sa_img{vertical-align:middle;margin:-.2em .4em .1em 0}.sa_2RowImg .sa_img{text-align:center;float:left}.sa_img img{max-width:55px;max-height:45px}.sa_sgDN .sa_tm{font-weight:bold}.sa_2RowImg,.sa_sgDN{color:#04c}.sa_sg strong,.sa_sgWH,.sa_sgWM,.sa_sgWU{color:#000}.sa_tmHS,.sa_tmHS strong{color:#639}.sa_sgPN{color:#737373}.sa_sgDN .sa_url{color:#568e1a;font-size:85%;border-bottom:1px solid #ededed;padding:.2em 0 .4em 0}.sa_om{background:#f5f5f5;border-top:1px solid #ccc;font-size:90%;height:1.6em;line-height:1.6em;padding:.16em 0 .2em 0}.sa_om a{color:#36b}.sa_om li{float:right;margin-left:1.3em}.sa_om ul{margin:0 .5em;padding:0}.sa_hv .sa_tm,.sa_om a:hover{text-decoration:underline}.sa_ghostbox{position:absolute;left:1px;z-index:-1}.sa_ghostbg{position:absolute;left:0;top:0;background:#fff;z-index:-2}.sa_sgDI,.sa_sgWM,.sa_sgWH,.sa_sgWU{padding-left:.55em}.sa_sgWM .sa_tm,.sa_sgWU .sa_tm{font-size:110%}.sa_ng{position:absolute;left:0;z-index:-1}.sa_sgNG{border-bottom:1px solid #ccc;margin-bottom:-.1em}.sa_sgFB{margin-top:0;position:relative}.sa_sgFB .sa_ds{margin-left:.35em}.sa_sgFB_img .sa_tm{margin-right:.2em}.sa_sgFB_img .sa_ds{margin-left:2em}.sa_sgFB .sa_img{margin:0}.sa_sgFB img{position:absolute;top:0}.sa_sgFB.sa_sg_first img{top:.3em}.sa_hv{background:#e5e5e5}.sa_sgID .sa_tm{font-family:Lucida Sans Unicode}.sa_sgID{*zoom:1;*margin-bottom:-8px}"), _w._G && _G.Mkt.toLowerCase() == "en-us" && _w.curUrl && curUrl != location.protocol + "//" + location.hostname + "/" && sj_ic(".sa_as{top:0px;}.sw_b3,.sa_2RowImg,.sa_om{border-color:#999;}.sw_b3{border-top:0;}");
sa_loc = {};
sa_loc["H_AS"] = "Web results";
sa_loc["H_AS_BA"] = "Suggestions";
sa_loc["H_EQ"] = "Also try these";
sa_loc["H_PN"] = "Popular now";
sa_loc["H_HS"] = "Recent searches";
sa_loc["L_ADS"] = "Advanced search";
sa_loc["L_DP"] = "Disable preview";
sa_loc["L_EP"] = "Enable preview";
sa_loc["L_H"] = "What's this?";
sa_loc["L_HS_OFF"] = "Turn history off";
sa_loc["L_HS_ON"] = "Turn history on";
sa_loc["L_MH"] = "Manage search history";
sa_loc["L_P"] = "Preferences";
sa_loc["L_SA"] = "See all results";
sa_loc["M_NR"] = "No suggestions matched your query.";
sa_loc["T_SQ"] = "Trending";
sa_loc["T_HS"] = "Recent";
