/*
YUI 3.12.0 (build 8655935)
Copyright 2013 Yahoo! Inc. All rights reserved.
Licensed under the BSD License.
http://yuilibrary.com/license/
*/

YUI.add("widget-base", function(e, t) {
    function R(e) {
        var t = this, n, r, i = t.constructor;
        t._strs = {}, t._cssPrefix = i.CSS_PREFIX || s(i.NAME.toLowerCase()), e = e || {}, R.superclass.constructor.call(t, e), r = t.get(T), r && (r !== P && (n = r), t.render(n))
    }
    var n = e.Lang, r = e.Node, i = e.ClassNameManager, s = i.getClassName, o, u = e.cached(function(e) {
        return e.substring(0, 1).toUpperCase() + e.substring(1)
    }), a = "content", f = "visible", l = "hidden", c = "disabled", h = "focused", p = "width", d = "height", v = "boundingBox", m = "contentBox", g = "parentNode", y = "ownerDocument", b = "auto", w = "srcNode", E = "body", S = "tabIndex", x = "id", T = "render", N = "rendered", C = "destroyed", k = "strings", L = "<div></div>", A = "Change", O = "loading", M = "_uiSet", _ = "", D = function() {}, P=!0, H=!1, B, j = {}, F = [f, c, d, p, h, S], I = e.UA.webkit, q = {};
    R.NAME = "widget", B = R.UI_SRC = "ui", R.ATTRS = j, j[x] = {
        valueFn: "_guid",
        writeOnce: P
    }, j[N] = {
        value: H,
        readOnly: P
    }, j[v] = {
        value: null,
        setter: "_setBB",
        writeOnce: P
    }, j[m] = {
        valueFn: "_defaultCB",
        setter: "_setCB",
        writeOnce: P
    }, j[S] = {
        value: null,
        validator: "_validTabIndex"
    }, j[h] = {
        value: H,
        readOnly: P
    }, j[c] = {
        value: H
    }, j[f] = {
        value: P
    }, j[d] = {
        value: _
    }, j[p] = {
        value: _
    }, j[k] = {
        value: {},
        setter: "_strSetter",
        getter: "_strGetter"
    }, j[T] = {
        value: H,
        writeOnce: P
    }, R.CSS_PREFIX = s(R.NAME.toLowerCase()), R.getClassName = function() {
        return s.apply(i, [R.CSS_PREFIX].concat(e.Array(arguments), !0))
    }, o = R.getClassName, R.getByNode = function(t) {
        var n, i = o();
        return t = r.one(t), t && (t = t.ancestor("." + i, !0), t && (n = q[e.stamp(t, !0)])), n || null
    }, e.extend(R, e.Base, {
        getClassName: function() {
            return s.apply(i, [this._cssPrefix].concat(e.Array(arguments), !0))
        },
        initializer: function(t) {
            var n = this.get(v);
            n instanceof r && this._mapInstance(e.stamp(n))
        },
        _mapInstance: function(e) {
            q[e] = this
        },
        destructor: function() {
            var t = this.get(v), n;
            t instanceof r && (n = e.stamp(t, !0), n in q && delete q[n], this._destroyBox())
        },
        destroy: function(e) {
            return this._destroyAllNodes = e, R.superclass.destroy.apply(this)
        },
        _destroyBox: function() {
            var e = this.get(v), t = this.get(m), n = this._destroyAllNodes, r;
            r = e && e.compareTo(t), this.UI_EVENTS && this._destroyUIEvents(), this._unbindUI(e), t && (n && t.empty(), t.remove(P)), r || (n && e.empty(), e.remove(P))
        },
        render: function(e) {
            return !this.get(C)&&!this.get(N) && (this.publish(T, {
                queuable: H,
                fireOnce: P,
                defaultTargetOnly: P,
                defaultFn: this._defRenderFn
            }), this.fire(T, {
                parentNode: e ? r.one(e): null
            })), this
        },
        _defRenderFn: function(e) {
            this._parentNode = e.parentNode, this.renderer(), this._set(N, P), this._removeLoadingClassNames()
        },
        renderer: function() {
            var e = this;
            e._renderUI(), e.renderUI(), e._bindUI(), e.bindUI(), e._syncUI(), e.syncUI()
        },
        bindUI: D,
        renderUI: D,
        syncUI: D,
        hide: function() {
            return this.set(f, H)
        },
        show: function() {
            return this.set(f, P)
        },
        focus: function() {
            return this._set(h, P)
        },
        blur: function() {
            return this._set(h, H)
        },
        enable: function() {
            return this.set(c, H)
        },
        disable: function() {
            return this.set(c, P)
        },
        _uiSizeCB: function(e) {
            this.get(m).toggleClass(o(a, "expanded"), e)
        },
        _renderBox: function(e) {
            var t = this, n = t.get(m), i = t.get(v), s = t.get(w), o = t.DEF_PARENT_NODE, u = s && s.get(y) || i.get(y) || n.get(y);
            s&&!s.compareTo(n)&&!n.inDoc(u) && s.replace(n), !i.compareTo(n.get(g))&&!i.compareTo(n) && (n.inDoc(u) && n.replace(i), i.appendChild(n)), e = e || o && r.one(o), e ? e.appendChild(i) : i.inDoc(u) || r.one(E).insert(i, 0)
        },
        _setBB: function(e) {
            return this._setBox(this.get(x), e, this.BOUNDING_TEMPLATE, !0)
        },
        _setCB: function(e) {
            return this.CONTENT_TEMPLATE === null ? this.get(v) : this._setBox(null, e, this.CONTENT_TEMPLATE, !1)
        },
        _defaultCB: function(e) {
            return this.get(w) || null
        },
        _setBox: function(t, n, i, s) {
            return n = r.one(n), n || (n = r.create(i), s ? this._bbFromTemplate=!0 : this._cbFromTemplate=!0), n.get(x) || n.set(x, t || e.guid()), n
        },
        _renderUI: function() {
            this._renderBoxClassNames(), this._renderBox(this._parentNode)
        },
        _renderBoxClassNames: function() {
            var e = this._getClasses(), t, n = this.get(v), r;
            n.addClass(o());
            for (r = e.length - 3; r >= 0; r--)
                t = e[r], n.addClass(t.CSS_PREFIX || s(t.NAME.toLowerCase()));
            this.get(m).addClass(this.getClassName(a))
        },
        _removeLoadingClassNames: function() {
            var e = this.get(v), t = this.get(m), n = this.getClassName(O), r = o(O);
            e.removeClass(r).removeClass(n), t.removeClass(r).removeClass(n)
        },
        _bindUI: function() {
            this._bindAttrUI(this._UI_ATTRS.BIND), this._bindDOM()
        },
        _unbindUI: function(e) {
            this._unbindDOM(e)
        },
        _bindDOM: function() {
            var t = this.get(v).get(y), n = R._hDocFocus;
            n || (n = R._hDocFocus = t.on("focus", this._onDocFocus, this), n.listeners = {
                count: 0
            }), n.listeners[e.stamp(this, !0)]=!0, n.listeners.count++, I && (this._hDocMouseDown = t.on("mousedown", this._onDocMouseDown, this))
        },
        _unbindDOM: function(t) {
            var n = R._hDocFocus, r = e.stamp(this, !0), i, s = this._hDocMouseDown;
            n && (i = n.listeners, i[r] && (delete i[r], i.count--), i.count === 0 && (n.detach(), R._hDocFocus = null)), I && s && s.detach()
        },
        _syncUI: function() {
            this._syncAttrUI(this._UI_ATTRS.SYNC)
        },
        _uiSetHeight: function(e) {
            this._uiSetDim(d, e), this._uiSizeCB(e !== _ && e !== b)
        },
        _uiSetWidth: function(e) {
            this._uiSetDim(p, e)
        },
        _uiSetDim: function(e, t) {
            this.get(v).setStyle(e, n.isNumber(t) ? t + this.DEF_UNIT : t)
        },
        _uiSetVisible: function(e) {
            this.get(v).toggleClass(this.getClassName(l), !e)
        },
        _uiSetDisabled: function(e) {
            this.get(v).toggleClass(this.getClassName(c), e)
        },
        _uiSetFocused: function(e, t) {
            var n = this.get(v);
            n.toggleClass(this.getClassName(h), e), t !== B && (e ? n.focus() : n.blur())
        },
        _uiSetTabIndex: function(e) {
            var t = this.get(v);
            n.isNumber(e) ? t.set(S, e) : t.removeAttribute(S)
        },
        _onDocMouseDown: function(e) {
            this._domFocus && this._onDocFocus(e)
        },
        _onDocFocus: function(e) {
            var t = R.getByNode(e.target), n = R._active;
            n && n !== t && (n._domFocus=!1, n._set(h, !1, {
                src: B
            }), R._active = null), t && (t._domFocus=!0, t._set(h, !0, {
                src: B
            }), R._active = t)
        },
        toString: function() {
            return this.name + "[" + this.get(x) + "]"
        },
        DEF_UNIT: "px",
        DEF_PARENT_NODE: null,
        CONTENT_TEMPLATE: L,
        BOUNDING_TEMPLATE: L,
        _guid: function() {
            return e.guid()
        },
        _validTabIndex: function(e) {
            return n.isNumber(e) || n.isNull(e)
        },
        _bindAttrUI: function(e) {
            var t, n = e.length;
            for (t = 0; t < n; t++)
                this.after(e[t] + A, this._setAttrUI)
        },
        _syncAttrUI: function(e) {
            var t, n = e.length, r;
            for (t = 0; t < n; t++)
                r = e[t], this[M + u(r)](this.get(r))
        },
        _setAttrUI: function(e) {
            e.target === this && this[M + u(e.attrName)](e.newVal, e.src)
        },
        _strSetter: function(t) {
            return e.merge(this.get(k), t)
        },
        getString: function(e) {
            return this
            .get(k)[e]
        },
        getStrings: function() {
            return this.get(k)
        },
        _UI_ATTRS: {
            BIND: F,
            SYNC: F
        }
    }), e.Widget = R
}, "3.12.0", {
    requires: ["attribute", "base-base", "base-pluginhost", "classnamemanager", "event-focus", "node-base", "node-style"],
    skinnable: !0
});
/*
YUI 3.12.0 (build 8655935)
Copyright 2013 Yahoo! Inc. All rights reserved.
Licensed under the BSD License.
http://yuilibrary.com/license/
*/

YUI.add("widget-stdmod", function(e, t) {
    function H(e) {}
    var n = e.Lang, r = e.Node, i = e.UA, s = e.Widget, o = "", u = "hd", a = "bd", f = "ft", l = "header", c = "body", h = "footer", p = "fillHeight", d = "stdmod", v = "Node", m = "Content", g = "firstChild", y = "childNodes", b = "ownerDocument", w = "contentBox", E = "height", S = "offsetHeight", x = "auto", T = "headerContentChange", N = "bodyContentChange", C = "footerContentChange", k = "fillHeightChange", L = "heightChange", A = "contentUpdate", O = "renderUI", M = "bindUI", _ = "syncUI", D = "_applyParsedConfig", P = e.Widget.UI_SRC;
    H.HEADER = l, H.BODY = c, H.FOOTER = h, H.AFTER = "after", H.BEFORE = "before", H.REPLACE = "replace";
    var B = H.HEADER, j = H.BODY, F = H.FOOTER, I = B + m, q = F + m, R = j + m;
    H.ATTRS = {
        headerContent: {
            value: null
        },
        footerContent: {
            value: null
        },
        bodyContent: {
            value: null
        },
        fillHeight: {
            value: H.BODY,
            validator: function(e) {
                return this._validateFillHeight(e)
            }
        }
    }, H.HTML_PARSER = {
        headerContent: function(e) {
            return this._parseStdModHTML(B)
        },
        bodyContent: function(e) {
            return this._parseStdModHTML(j)
        },
        footerContent: function(e) {
            return this._parseStdModHTML(F)
        }
    }, H.SECTION_CLASS_NAMES = {
        header: s.getClassName(u),
        body: s.getClassName(a),
        footer: s.getClassName(f)
    }, H.TEMPLATES = {
        header: '<div class="' + H.SECTION_CLASS_NAMES[B] + '"></div>',
        body: '<div class="' + H.SECTION_CLASS_NAMES[j] + '"></div>',
        footer: '<div class="' + H.SECTION_CLASS_NAMES[F] + '"></div>'
    }, H.prototype = {
        initializer: function() {
            this._stdModNode = this.get(w), e.before(this._renderUIStdMod, this, O), e.before(this._bindUIStdMod, this, M), e.before(this._syncUIStdMod, this, _)
        },
        _syncUIStdMod: function() {
            var e = this._stdModParsed;
            (!e ||!e[I]) && this._uiSetStdMod(B, this.get(I)), (!e ||!e[R]) && this._uiSetStdMod(j, this.get(R)), (!e ||!e[q]) && this._uiSetStdMod(F, this.get(q)), this._uiSetFillHeight(this.get(p))
        },
        _renderUIStdMod: function() {
            this._stdModNode.addClass(s.getClassName(d)), this._renderStdModSections(), this.after(T, this._afterHeaderChange), this.after(N, this._afterBodyChange), this.after(C, this._afterFooterChange)
        },
        _renderStdModSections: function() {
            n.isValue(this.get(I)) && this._renderStdMod(B), n.isValue(this.get(R)) && this._renderStdMod(j), n.isValue(this.get(q)) && this._renderStdMod(F)
        },
        _bindUIStdMod: function() {
            this.after(k, this._afterFillHeightChange), this.after(L, this._fillHeight), this.after(A, this._fillHeight)
        },
        _afterHeaderChange: function(e) {
            e.src !== P && this._uiSetStdMod(B, e.newVal, e.stdModPosition)
        },
        _afterBodyChange: function(e) {
            e.src !== P && this._uiSetStdMod(j, e.newVal, e.stdModPosition)
        },
        _afterFooterChange: function(e) {
            e.src !== P && this._uiSetStdMod(F, e.newVal, e.stdModPosition)
        },
        _afterFillHeightChange: function(e) {
            this._uiSetFillHeight(e.newVal)
        },
        _validateFillHeight: function(e) {
            return !e || e == H.BODY || e == H.HEADER || e == H.FOOTER
        },
        _uiSetFillHeight: function(e) {
            var t = this.getStdModNode(e), n = this._currFillNode;
            n && t !== n && n.setStyle(E, o), t && (this._currFillNode = t), this._fillHeight()
        },
        _fillHeight: function() {
            if (this.get(p)) {
                var e = this.get(E);
                e != o && e != x && this.fillHeight(this.getStdModNode(this.get(p)))
            }
        },
        _uiSetStdMod: function(e, t, r) {
            if (n.isValue(t)) {
                var i = this.getStdModNode(e, !0);
                this._addStdModContent(i, t, r), this.set(e + m, this._getStdModContent(e), {
                    src: P
                })
            } else 
                this._eraseStdMod(e);
            this.fire(A)
        },
        _renderStdMod: function(e) {
            var t = this.get(w), n = this._findStdModSection(e);
            return n || (n = this._getStdModTemplate(e)), this._insertStdModSection(t, e, n), this[e + v] = n, this[e + v]
        },
        _eraseStdMod: function(e) {
            var t = this.getStdModNode(e);
            t && (t.remove(!0), delete this[e + v])
        },
        _insertStdModSection: function(e, t, n) {
            var r = e.get(g);
            if (t === F ||!r)
                e.appendChild(n);
            else if (t === B)
                e.insertBefore(n, r);
            else {
                var i = this[F + v];
                i ? e.insertBefore(n, i) : e.appendChild(n)
            }
        },
        _getStdModTemplate: function(e) {
            return r.create(H.TEMPLATES[e], this._stdModNode.get(b))
        },
        _addStdModContent: function(e, t, n) {
            switch (n) {
            case H.BEFORE:
                n = 0;
                break;
            case H.AFTER:
                n = undefined;
                break;
            default:
                n = H.REPLACE
            }
            e.insert(t, n)
        },
        _getPreciseHeight: function(e) {
            var t = e ? e.get(S): 0, n = "getBoundingClientRect";
            if (e && e.hasMethod(n)) {
                var r = e.invoke(n);
                r && (t = r.bottom - r.top)
            }
            return t
        },
        _findStdModSection: function(e) {
            return this.get(w).one("> ." + H.SECTION_CLASS_NAMES[e])
        },
        _parseStdModHTML: function(t) {
            var n = this._findStdModSection(t);
            return n ? (this._stdModParsed || (this._stdModParsed = {}, e.before(this._applyStdModParsedConfig, this, D)), this._stdModParsed[t + m] = 1, n.get("innerHTML")) : null
        },
        _applyStdModParsedConfig: function(e, t, n) {
            var r = this._stdModParsed;
            r && (r[I]=!(I in t) && I in r, r[R]=!(R in t) && R in r, r[q]=!(q in t) && q in r)
        },
        _getStdModContent: function(e) {
            return this[e + v] ? this[e + v].get(y) : null
        },
        setStdModContent: function(e, t, n) {
            this.set(e + m, t, {
                stdModPosition: n
            })
        },
        getStdModNode: function(e, t) {
            var n = this[e + v] || null;
            return !n && t && (n = this._renderStdMod(e)), n
        },
        fillHeight: function(e) {
            if (e) {
                var t = this.get(w), r = [this.headerNode, this.bodyNode, this.footerNode], s, o, u = 0, a = 0, f=!1;
                for (var l = 0, c = r.length; l < c; l++)
                    s = r[l], s && (s !== e ? u += this._getPreciseHeight(s) : f=!0);
                f && ((i.ie || i.opera) && e.set(S, 0), o = t.get(S) - parseInt(t.getComputedStyle("paddingTop"), 10) - parseInt(t.getComputedStyle("paddingBottom"), 10) - parseInt(t.getComputedStyle("borderBottomWidth"), 10) - parseInt(t.getComputedStyle("borderTopWidth"), 10), n.isNumber(o) && (a = o - u, a >= 0 && e.set(S, a)))
            }
        }
    }, e.WidgetStdMod = H
}, "3.12.0", {
    requires: ["base-build", "widget"]
});
/*
YUI 3.12.0 (build 8655935)
Copyright 2013 Yahoo! Inc. All rights reserved.
Licensed under the BSD License.
http://yuilibrary.com/license/
*/

YUI.add("widget-stack", function(e, t) {
    function O(e) {}
    var n = e.Lang, r = e.UA, i = e.Node, s = e.Widget, o = "zIndex", u = "shim", a = "visible", f = "boundingBox", l = "renderUI", c = "bindUI", h = "syncUI", p = "offsetWidth", d = "offsetHeight", v = "parentNode", m = "firstChild", g = "ownerDocument", y = "width", b = "height", w = "px", E = "shimdeferred", S = "shimresize", x = "visibleChange", T = "widthChange", N = "heightChange", C = "shimChange", k = "zIndexChange", L = "contentUpdate", A = "stacked";
    O.ATTRS = {
        shim: {
            value: r.ie == 6
        },
        zIndex: {
            value: 0,
            setter: "_setZIndex"
        }
    }, O.HTML_PARSER = {
        zIndex: function(e) {
            return this._parseZIndex(e)
        }
    }, O.SHIM_CLASS_NAME = s.getClassName(u), O.STACKED_CLASS_NAME = s.getClassName(A), O.SHIM_TEMPLATE = '<iframe class="' + O.SHIM_CLASS_NAME + '" frameborder="0" title="Widget Stacking Shim" src="javascript:false" tabindex="-1" role="presentation"></iframe>', O.prototype = {
        initializer: function() {
            this._stackNode = this.get(f), this._stackHandles = {}, e.after(this._renderUIStack, this, l), e.after(this._syncUIStack, this, h), e.after(this._bindUIStack, this, c)
        },
        _syncUIStack: function() {
            this._uiSetShim(this.get(u)), this._uiSetZIndex(this.get(o))
        },
        _bindUIStack: function() {
            this.after(C, this._afterShimChange), this.after(k, this._afterZIndexChange)
        },
        _renderUIStack: function() {
            this._stackNode.addClass(O.STACKED_CLASS_NAME)
        },
        _parseZIndex: function(e) {
            var t;
            return !e.inDoc() || e.getStyle("position") === "static" ? t = "auto" : t = e.getComputedStyle("zIndex"), t === "auto" ? null : t
        },
        _setZIndex: function(e) {
            return n.isString(e) && (e = parseInt(e, 10)), n.isNumber(e) || (e = 0), e
        },
        _afterShimChange: function(e) {
            this._uiSetShim(e.newVal)
        },
        _afterZIndexChange: function(e) {
            this._uiSetZIndex(e.newVal)
        },
        _uiSetZIndex: function(e) {
            this._stackNode.setStyle(o, e)
        },
        _uiSetShim: function(e) {
            e ? (this.get(a) ? this._renderShim() : this._renderShimDeferred(), r.ie == 6 && this._addShimResizeHandlers()) : this._destroyShim()
        },
        _renderShimDeferred: function() {
            this._stackHandles[E] = this._stackHandles[E] || [];
            var e = this._stackHandles[E], t = function(e) {
                e.newVal && this._renderShim()
            };
            e.push(this.on(x, t))
        },
        _addShimResizeHandlers: function() {
            this._stackHandles[S] = this._stackHandles[S] || [];
            var e = this.sizeShim, t = this._stackHandles[S];
            t.push(this.after(x, e)), t.push(this.after(T, e)), t.push(this.after(N, e)), t.push(this.after(L, e))
        },
        _detachStackHandles: function(e) {
            var t = this._stackHandles[e], n;
            if (t && t.length > 0)
                while (n = t.pop())
                    n.detach()
        },
        _renderShim: function() {
            var e = this._shimNode, t = this._stackNode;
            e || (e = this._shimNode = this._getShimTemplate(), t.insertBefore(e, t.get(m)), this._detachStackHandles(E), this.sizeShim())
        },
        _destroyShim: function() {
            this._shimNode && (this._shimNode.get(v).removeChild(this._shimNode), this._shimNode = null, this._detachStackHandles(E), this._detachStackHandles(S))
        },
        sizeShim: function() {
            var e = this._shimNode, t = this._stackNode;
            e && r.ie === 6 && this.get(a) && (e.setStyle(y, t.get(p) + w), e.setStyle(b, t.get(d) + w))
        },
        _getShimTemplate: function() {
            return i.create(O.SHIM_TEMPLATE, this._stackNode.get(g))
        }
    }, e.WidgetStack = O
}, "3.12.0", {
    requires: ["base-build", "widget"],
    skinnable: !0
});
/*
YUI 3.12.0 (build 8655935)
Copyright 2013 Yahoo! Inc. All rights reserved.
Licensed under the BSD License.
http://yuilibrary.com/license/
*/

YUI.add("widget-position-constrain", function(e, t) {
    function m(e) {}
    var n = "constrain", r = "constrain|xyChange", i = "constrainChange", s = "preventOverlap", o = "align", u = "", a = "bindUI", f = "xy", l = "x", c = "y", h = e.Node, p = "viewportRegion", d = "region", v;
    m.ATTRS = {
        constrain: {
            value: null,
            setter: "_setConstrain"
        },
        preventOverlap: {
            value: !1
        }
    }, v = m._PREVENT_OVERLAP = {
        x: {
            tltr: 1,
            blbr: 1,
            brbl: 1,
            trtl: 1
        },
        y: {
            trbr: 1,
            tlbl: 1,
            bltl: 1,
            brtr: 1
        }
    }, m.prototype = {
        initializer: function() {
            this._posNode || e.error("WidgetPosition needs to be added to the Widget, before WidgetPositionConstrain is added"), e.after(this._bindUIPosConstrained, this, a)
        },
        getConstrainedXY: function(e, t) {
            t = t || this.get(n);
            var r = this._getRegion(t===!0 ? null : t), i = this._posNode.get(d);
            return [this._constrain(e[0], l, i, r), this._constrain(e[1], c, i, r)]
        },
        constrain: function(e, t) {
            var r, i, s = t || this.get(n);
            s && (r = e || this.get(f), i = this.getConstrainedXY(r, s), (i[0] !== r[0] || i[1] !== r[1]) && this.set(f, i, {
                constrained: !0
            }))
        },
        _setConstrain: function(e) {
            return e===!0 ? e : h.one(e)
        },
        _constrain: function(e, t, n, r) {
            if (r) {
                this.get(s) && (e = this._preventOverlap(e, t, n, r));
                var i = t == l, o = i ? r.width: r.height, u = i ? n.width: n.height, a = i ? r.left: r.top, f = i ? r.right - u: r.bottom - u;
                if (e < a || e > f)
                    u < o ? e < a ? e = a : e > f && (e = f) : e = a
            }
            return e
        },
        _preventOverlap: function(e, t, n, r) {
            var i = this.get(o), s = t === l, a, f, c, h, p, d;
            return i && i.points && v[t][i.points.join(u)] && (f = this._getRegion(i.node), f && (a = s ? n.width : n.height, c = s ? f.left : f.top, h = s ? f.right : f.bottom, p = s ? f.left - r.left : f.top - r.top, d = s ? r.right - f.right : r.bottom - f.bottom), e > c ? d < a && p > a && (e = c - a) : p < a && d > a && (e = h)), e
        },
        _bindUIPosConstrained: function() {
            this.after(i, this._afterConstrainChange), this._enableConstraints(this.get(n))
        },
        _afterConstrainChange: function(e) {
            this._enableConstraints(e.newVal)
        },
        _enableConstraints: function(e) {
            e ? (this.constrain(), this._cxyHandle = this._cxyHandle || this.on(r, this._constrainOnXYChange)) : this._cxyHandle && (this._cxyHandle.detach(), this._cxyHandle = null)
        },
        _constrainOnXYChange: function(e) {
            e.constrained || (e.newVal = this.getConstrainedXY(e.newVal))
        },
        _getRegion: function(e) {
            var t;
            return e ? (e = h.one(e), e && (t = e.get(d))) : t = this._posNode.get(p), t
        }
    }, e.WidgetPositionConstrain = m
}, "3.12.0", {
    requires: ["widget-position"]
});
/*
YUI 3.12.0 (build 8655935)
Copyright 2013 Yahoo! Inc. All rights reserved.
Licensed under the BSD License.
http://yuilibrary.com/license/
*/

YUI.add("overlay", function(e, t) {
    e.Overlay = e.Base.create("overlay", e.Widget, [e.WidgetStdMod, e.WidgetPosition, e.WidgetStack, e.WidgetPositionAlign, e.WidgetPositionConstrain])
}, "3.12.0", {
    requires: ["widget", "widget-stdmod", "widget-position", "widget-position-align", "widget-stack", "widget-position-constrain"],
    skinnable: !0
});
/*
YUI 3.12.0 (build 8655935)
Copyright 2013 Yahoo! Inc. All rights reserved.
Licensed under the BSD License.
http://yuilibrary.com/license/
*/

YUI.add("swfdetect", function(e, t) {
    function c(e) {
        return parseInt(e, 10)
    }
    function h(e) {
        i.isNumber(c(e[0])) && (r.flashMajor = e[0]), i.isNumber(c(e[1])) && (r.flashMinor = e[1]), i.isNumber(c(e[2])) && (r.flashRev = e[2])
    }
    var n = 0, r = e.UA, i = e.Lang, s = "ShockwaveFlash", o, u, a, f, l;
    if (r.gecko || r.webkit || r.opera) {
        if (o = navigator.mimeTypes["application/x-shockwave-flash"])
            if (u = o.enabledPlugin)
                a = u.description.replace(/\s[rd]/g, ".").replace(/[A-Za-z\s]+/g, "").split("."), h(a)
    } else if (r.ie) {
        try {
            f = new ActiveXObject(s + "." + s + ".6"), f.AllowScriptAccess = "always"
        } catch (p) {
            f !== null && (n = 6)
        }
        if (n === 0)
            try {
                l = new ActiveXObject(s + "." + s), a = l.GetVariable("$version").replace(/[A-Za-z\s]+/g, "").split(","), h(a)
            } catch (d) {}
    }
    e.SWFDetect = {
        getFlashVersion: function() {
            return String(r.flashMajor) + "." + String(r.flashMinor) + "." + String(r.flashRev)
        },
        isFlashVersionAtLeast: function(e, t, n) {
            var i = c(r.flashMajor), s = c(r.flashMinor), o = c(r.flashRev);
            return e = c(e || 0), t = c(t || 0), n = c(n || 0), e === i ? t === s ? n <= o : t < s : e < i
        }
    }
}, "3.12.0", {
    requires: ["yui-base"]
});
/*
YUI 3.12.0 (build 8655935)
Copyright 2013 Yahoo! Inc. All rights reserved.
Licensed under the BSD License.
http://yuilibrary.com/license/
*/

YUI.add("swf", function(e, t) {
    function d(t, n, d) {
        this._id = e.guid("yuiswf");
        var v = this._id, m = o.one(t), d = d || {}, g = d.version || l, y = (g + "").split("."), b = r.isFlashVersionAtLeast(parseInt(y[0], 10), parseInt(y[1], 10), parseInt(y[2], 10)), w = r.isFlashVersionAtLeast(8, 0, 0), E = w&&!b && d.useExpressInstall, S = E ? c: n, x = "<object ", T, N, C = "yId=" + e.id + "&YUISwfId=" + v + "&YUIBridgeCallback=" + h + "&allowedDomain=" + document.location.hostname;
        e.SWF._instances[v] = this;
        if (m && (b || E) && S) {
            x += 'id="' + v + '" ', s.ie ? x += 'classid="' + a + '" ' : x += 'type="' + f + '" data="' + u.html(S) + '" ', T = "100%", N = "100%", x += 'width="' + T + '" height="' + N + '">', s.ie && (x += '<param name="movie" value="' + u.html(S) + '"/>');
            for (var k in d.fixedAttributes)
                p.hasOwnProperty(k) && (x += '<param name="' + u.html(k) + '" value="' + u.html(d.fixedAttributes[k]) + '"/>');
            for (var L in d.flashVars) {
                var A = d.flashVars[L];
                i.isString(A) && (C += "&" + u.html(L) + "=" + u.html(encodeURIComponent(A)))
            }
            C && (x += '<param name="flashVars" value="' + C + '"/>'), x += "</object>", m.set("innerHTML", x), this._swf = o.one("#" + v)
        } else {
            var O = {};
            O.type = "wrongflashversion", this.publish("wrongflashversion", {
                fireOnce: !0
            }), this.fire("wrongflashversion", O)
        }
    }
    var n = e.Event, r = e.SWFDetect, i = e.Lang, s = e.UA, o = e.Node, u = e.Escape, a = "clsid:d27cdb6e-ae6d-11cf-96b8-444553540000", f = "application/x-shockwave-flash", l = "10.0.22", c = "http://fpdownload.macromedia.com/pub/flashplayer/update/current/swf/autoUpdater.swf?" + Math.random(), h = "SWF.eventHandler", p = {
        align: "",
        allowFullScreen: "",
        allowNetworking: "",
        allowScriptAccess: "",
        base: "",
        bgcolor: "",
        loop: "",
        menu: "",
        name: "",
        play: "",
        quality: "",
        salign: "",
        scale: "",
        tabindex: "",
        wmode: ""
    };
    d._instances = d._instances || {}, d.eventHandler = function(e, t) {
        d._instances[e]._eventHandler(t)
    }, d.prototype = {
        _eventHandler: function(e) {
            e.type === "swfReady" ? (this.publish("swfReady", {
                fireOnce: !0
            }), this.fire("swfReady", e)) : e.type !== "log" && this.fire(e.type, e)
        },
        callSWF: function(e, t) {
            return t || (t = []), this._swf._node[e] ? this._swf._node[e].apply(this._swf._node, t) : null
        },
        toString: function() {
            return "SWF " + this._id
        }
    }, e.augment(d, e.EventTarget), e.SWF = d
}, "3.12.0", {
    requires: ["event-custom", "node", "swfdetect", "escape"]
});
/*
YUI 3.12.0 (build 8655935)
Copyright 2013 Yahoo! Inc. All rights reserved.
Licensed under the BSD License.
http://yuilibrary.com/license/
*/

YUI.add("lang/autocomplete-list_en", function(e) {
    e.Intl.add("autocomplete-list", "en", {
        item_selected: "{item} selected.",
        items_available: "Suggestions are available. Use up and down arrows to select."
    })
}, "3.12.0");
/*
YUI 3.12.0 (build 8655935)
Copyright 2013 Yahoo! Inc. All rights reserved.
Licensed under the BSD License.
http://yuilibrary.com/license/
*/

YUI.add("autocomplete-list", function(e, t) {
    var n = e.Lang, r = e.Node, i = e.Array, s = e.UA.ie && e.UA.ie < 7, o = 9, u = "_CLASS_ITEM", a = "_CLASS_ITEM_ACTIVE", f = "_CLASS_ITEM_HOVER", l = "_SELECTOR_ITEM", c = "activeItem", h = "alwaysShowList", p = "circular", d = "hoveredItem", v = "id", m = "item", g = "list", y = "result", b = "results", w = "visible", E = "width", S = "select", x = e.Base.create("autocompleteList", e.Widget, [e.AutoCompleteBase, e.WidgetPosition, e.WidgetPositionAlign], {
        ARIA_TEMPLATE: "<div/>",
        ITEM_TEMPLATE: "<li/>",
        LIST_TEMPLATE: "<ul/>",
        UI_EVENTS: function() {
            var t = e.merge(e.Node.DOM_EVENTS);
            return delete t.valuechange, delete t.valueChange, t
        }(),
        initializer: function() {
            var t = this.get("inputNode");
            if (!t) {
                e.error("No inputNode specified.");
                return 
            }
            this._inputNode = t, this._listEvents = [], this.DEF_PARENT_NODE = t.get("parentNode"), this[u] = this.getClassName(m), this[a] = this.getClassName(m, "active"), this[f] = this.getClassName(m, "hover"), this[l] = "." + this[u], this.publish(S, {
                defaultFn: this._defSelectFn
            })
        },
        destructor: function() {
            while (this._listEvents.length)
                this._listEvents.pop().detach();
            this._ariaNode && this._ariaNode.remove().destroy(!0)
        },
        bindUI: function() {
            this._bindInput(), this._bindList()
        },
        renderUI: function() {
            var t = this._createAriaNode(), n = this.get("boundingBox"), r = this.get("contentBox"), i = this._inputNode, o = this._createListNode(), u = i.get("parentNode");
            i.addClass(this.getClassName("input")).setAttrs({
                "aria-autocomplete": g,
                "aria-expanded": !1,
                "aria-owns": o.get("id")
            }), u.append(t), s && n.plug(e.Plugin.Shim), this._ariaNode = t, this._boundingBox = n, this._contentBox = r, this._listNode = o, this._parentNode = u
        },
        syncUI: function() {
            this._syncResults(), this._syncVisibility()
        },
        hide: function() {
            return this.get(h) ? this : this.set(w, !1)
        },
        selectItem: function(e, t) {
            if (e) {
                if (!e.hasClass(this[u]))
                    return this
            } else {
                e = this.get(c);
                if (!e)
                    return this
            }
            return this.fire(S, {
                itemNode: e,
                originEvent: t || null,
                result: e.getData(y)
            }), this
        },
        _activateNextItem: function() {
            var e = this.get(c), t;
            return e ? t = e.next(this[l]) || (this.get(p) ? null : e) : t = this._getFirstItemNode(), this.set(c, t), this
        },
        _activatePrevItem: function() {
            var e = this.get(c), t = e ? e.previous(this[l]): this.get(p) && this._getLastItemNode();
            return this.set(c, t || null), this
        },
        _add: function(t) {
            var r = [];
            return i.each(n.isArray(t) ? t : [t], function(e) {
                r.push(this._createItemNode(e).setData(y, e))
            }, this), r = e.all(r), this._listNode.append(r.toFrag()), r
        },
        _ariaSay: function(e, t) {
            var r = this.get("strings." + e);
            this._ariaNode.set("text", t ? n.sub(r, t) : r)
        },
        _bindInput: function() {
            var e = this._inputNode, t, n, r;
            this.get("align") === null && (r = this.get("tokenInput"), t = r && r.get("boundingBox") || e, this.set("align", {
                node: t,
                points: ["tl", "bl"]
            }), !this.get(E) && (n = t.get("offsetWidth")) && this.set(E, n)), this._listEvents = this._listEvents.concat([e.after("blur", this._afterListInputBlur, this), e.after("focus", this._afterListInputFocus, this)])
        },
        _bindList: function() {
            this._listEvents = this._listEvents.concat([e.one("doc").after("click", this._afterDocClick, this), e.one("win").after("windowresize", this._syncPosition, this), this.after({
                mouseover: this._afterMouseOver,
                mouseout: this._afterMouseOut,
                activeItemChange: this._afterActiveItemChange,
                alwaysShowListChange: this._afterAlwaysShowListChange,
                hoveredItemChange: this._afterHoveredItemChange,
                resultsChange: this._afterResultsChange,
                visibleChange: this._afterVisibleChange
            }), this._listNode.delegate("click", this._onItemClick, this[l], this)])
        },
        _clear: function() {
            this.set(c, null), this._set(d, null), this._listNode.get("children").remove(!0)
        },
        _createAriaNode: function() {
            var e = r.create(this.ARIA_TEMPLATE);
            return e.addClass(this.getClassName("aria")).setAttrs({
                "aria-live": "polite",
                role: "status"
            })
        },
        _createItemNode: function(t) {
            var n = r.create(this.ITEM_TEMPLATE);
            return n.addClass(this[u]).setAttrs({
                id: e.stamp(n),
                role: "option"
            }).setAttribute("data-text", t.text).append(t.display)
        },
        _createListNode: function() {
            var t = this.get("listNode") || r.create(this.LIST_TEMPLATE);
            return t.addClass(this.getClassName(g)).setAttrs({
                id: e.stamp(t),
                role: "listbox"
            }), this._set("listNode", t), this.get("contentBox").append(t), t
        },
        _getFirstItemNode: function() {
            return this._listNode.one(this[l])
        },
        _getLastItemNode: function() {
            return this._listNode.one(this[l] + ":last-child")
        },
        _syncPosition: function() {
            this._syncUIPosAlign(), this._syncShim()
        },
        _syncResults: function(e) {
            e || (e = this.get(b)), this._clear(), e.length && (this._add(e), this._ariaSay("items_available")), this._syncPosition(), this.get("activateFirstItem")&&!this.get(c) && this.set(c, this._getFirstItemNode())
        },
        _syncShim: s ? function() {
            var e = this._boundingBox.shim;
            e && e.sync()
        }
        : function() {},
        _syncVisibility: function(t) {
            this.get(h) && (t=!0, this.set(w, t)), typeof t == "undefined" && (t = this.get(w)), this._inputNode.set("aria-expanded", t), this._boundingBox.set("aria-hidden", !t), t ? this._syncPosition() : (this.set(c, null), this._set(d, null), this._boundingBox.get("offsetWidth")), e.UA.ie === 7 && e.one("body").addClass("yui3-ie7-sucks").removeClass("yui3-ie7-sucks")
        },
        _afterActiveItemChange: function(t) {
            var n = this._inputNode, r = t.newVal, i = t.prevVal, s;
            i && i._node && i.removeClass(this[a]), r ? (r.addClass(this[a]), n.set("aria-activedescendant", r.get(v))) : n.removeAttribute("aria-activedescendant"), this.get("scrollIntoView") && (s = r || n, (!s.inRegion(e.DOM.viewportRegion(), !0) ||!s.inRegion(this._contentBox, !0)) && s.scrollIntoView())
        },
        _afterAlwaysShowListChange: function(e) {
            this.set(w, e.newVal || this.get(b).length > 0)
        },
        _afterDocClick: function(e) {
            var t = this._boundingBox, n = e.target;
            n !== this._inputNode && n !== t && n.ancestor("#" + t.get("id"), !0) && this.hide()
        },
        _afterHoveredItemChange: function(e) {
            var t = e.newVal, n = e.prevVal;
            n && n.removeClass(this[f]), t && t.addClass(this[f])
        },
        _afterListInputBlur: function() {
            this._listInputFocused=!1, this.get(w)&&!this._mouseOverList && (this._lastInputKey !== o ||!this.get("tabSelect") ||!this.get(c)) && this.hide()
        },
        _afterListInputFocus: function() {
            this._listInputFocused=!0
        },
        _afterMouseOver: function(e) {
            var t = e.domEvent.target.ancestor(this[l], !0);
            this._mouseOverList=!0, t && this._set(d, t)
        },
        _afterMouseOut: function() {
            this._mouseOverList=!1, this._set(d, null)
        },
        _afterResultsChange: function(e) {
            this._syncResults(e.newVal
            ), this.get(h) || this.set(w, !!e.newVal.length)
        },
        _afterVisibleChange: function(e) {
            this._syncVisibility(!!e.newVal)
        },
        _onItemClick: function(e) {
            var t = e.currentTarget;
            this.set(c, t), this.selectItem(t, e)
        },
        _defSelectFn: function(e) {
            var t = e.result.text;
            this._inputNode.focus(), this._updateValue(t), this._ariaSay("item_selected", {
                item: t
            }), this.hide()
        }
    }, {
        ATTRS: {
            activateFirstItem: {
                value: !1
            },
            activeItem: {
                setter: e.one,
                value: null
            },
            alwaysShowList: {
                value: !1
            },
            circular: {
                value: !0
            },
            hoveredItem: {
                readOnly: !0,
                value: null
            },
            listNode: {
                writeOnce: "initOnly",
                value: null
            },
            scrollIntoView: {
                value: !1
            },
            strings: {
                valueFn: function() {
                    return e.Intl.get("autocomplete-list")
                }
            },
            tabSelect: {
                value: !0
            },
            visible: {
                value: !1
            }
        },
        CSS_PREFIX: e.ClassNameManager.getClassName("aclist")
    });
    e.AutoCompleteList = x, e.AutoComplete = x
}, "3.12.0", {
    lang: ["en", "es", "hu", "it"],
    requires: ["autocomplete-base", "event-resize", "node-screen", "selector-css3", "shim-plugin", "widget", "widget-position", "widget-position-align"],
    skinnable: !0
});
YUI.add("stencil-gallery", function(e, t) {
    "use strict";
    function k(r) {
        this.constructor.superclass.constructor.call(this, r);
        if (this._node && this._node.get("tagName").toUpperCase() === "BODY")
            return;
        var i = this._node, s, o = i.all(y), u = i.one(g), p = i.one(b), v = i.ancestor("[dir]", !0), w = u && u.all(m), T = w && w.size(), N;
        if (!i ||!u ||!p)
            return;
        this.node = i, this.sliderEl = u, this.maskEl = p, this.currentWidth = p.get("offsetWidth"), this.itemsEl = u && this.sliderEl.all(m), this.indicatorsEl = o.size() > 0 ? o : n, this.prevEl = i.one(E), this.nextEl = i.one(S), this.itemsLength = T, this.dir = r.dir, this.cacheItemsPositions(), this.dir || (this.dir = v ? v.getAttribute("dir") : "ltr", this.dir = this.dir === "rtl"?-1 : 1), this.currentItem = r.currentItem || parseInt(i.getAttribute("data-currentItem"), 10) || u && w.indexOf(u.one("." + d)), this.currentItem < 0 && (this.currentItem = 0), x && x.touch && (s = i.getDOMNode(), N = (new Stencil.Swiper(i, {
            config: {
                swipe: {
                    acceptable_opposite_axis_movement: 300,
                    min_swipe_distance: 10
                }
            }
        })).captureTouch(), s.addEventListener(h, e.bind(this.swipe, this), !1), s.addEventListener(l, e.bind(this.swipe, this), !1), s.addEventListener(c, e.bind(this.swipe, this), !1), s.addEventListener(a, e.bind(this.swipe, this), !1), s.addEventListener(f, e.bind(this.swipe, this), !1), s.addEventListener("tap", function(e) {}, !1));
        if (!this.node ||!this.sliderEl ||!this.maskEl ||!this.itemsEl.size())
            return;
        o && this.updateIndicators(), this.updateToolbar(), i.once("mouseover", this.loadDelayedImages, this), this.resizeHandle = e.on(t + ":resize", this.onResize, this)
    }
    var n = null, r=!0, i=!1, s = "prev", o = "next", u = "jump-to", a = "swipe", f = "slide", l = "init", c = "end", h = "dragstart", p = ".Gallery-", d = "Selected", v = "Active", m = p + "Item", g = p + "Slider", y = p + "Indicator", b = p + "Mask", w = "Disabled", E = "button[data-jump-to=prev]", S = "button[data-jump-to=next]", x = Modernizr, T = e.one("body"), N = "disabled", C;
    e.one(e.config.win).on("resize", function() {
        clearTimeout(C), C = setTimeout(function() {
            e.fire(t + ":resize")
        }, 100)
    }, this), e.extend(k, e.Stencil.BasePlugin, {
        currentItem: 0,
        itemsLength: 0,
        itemsPositions: [],
        itemsWidths: [],
        currentWidth: 0,
        totalWidth: 0,
        isDelayedLoaded: i,
        destroy: function() {
            this.node = n, this.sliderEl = n, this.maskEl = n, this.itemsEl = n, this.indicatorsEl = n, this.itemsPositions = n, this.prevEl = n, this.nextEl = n, this.resizeHandle && (this.resizeHandle.detach(), this.resizeHandle = n), this.constructor.superclass.destroy.apply(this, arguments)
        },
        loadDelayedImages: function() {
            T.fire("imageloader:delayed", this.node), this.isDelayedLoaded = r
        },
        cacheItemsPositions: function() {
            var e = [0], t = [], n = 0;
            this.itemsEl.each(function(r, i) {
                var s = r.get("children").item(0), o;
                i !== 0 && e.push(e[i - 1] + t[i - 1]), s && s.get("tagName").toUpperCase() === "IMG" ? o = parseInt(s.get("width"), 10) : o = parseInt(s.get("offsetWidth"), 10), t.push(o), n += o
            }, this), this.totalWidth = n, this.itemsWidths = t, this.itemsPositions = e
        },
        jumpTo: function(e, t) {
            var n = this.currentWidth, r = this.currentItem, u = this.itemsLength, a = this.itemsWidths, f = this.itemsPositions, l = this.totalWidth, c = 0;
            e = typeof e == "number" || typeof e == "string" ? e : r, t = t || "slide";
            if (e === s) {
                while (r !== 0 && f[r - 1] > l - n)
                    r -= 1;
                if (r !== 0)
                    do 
                        r -= 1, c += a[r];
                while (r !== 0 && c + a[r - 1] < n)
                } else if (e === o) {
                    if (r + 1 !== u)
                        do 
                            c += a[r], r += 1;
                            while (r + 1 !== u && c + a[r + 1] < n)
                            } else 
                                r = parseInt(e, 10);
                                r !== 0 && f[r] + n >= l && (r = u - 1), this.currentItem = r, t === "slide" ? this.slide(r) : t === "none" && this.translate(this.sliderEl, - this.itemsPositions[e], 0, i), this.updateToolbar(), this.updateIndicators()
            }, slide:
        function(e) {
            this.translate(this.sliderEl, - this.itemsPositions[e], 0)
        }, translate:
        function(e, t, n, r) {
            var s = {}, o = this.dir;
            return t = t ? Math.max(t, this.currentWidth - this.totalWidth) : t, t = t || t === 0 ? o * t + "px" : "", n = n || n === 0 ? o * n + "px" : "", x && x.csstransforms3d ? s.transform = t || n ? "translate3d(" + t + "," + n + ",0)" : "" : (s.left = t, s.top = n), r === i ||!t&&!n&&!r ? (e.setStyles(s), "notransition") : (s.duration = .4, e.transition(s, r), "transition")
        }, handleEvent:
        function(e) {
            var t = e.target.ancestor("[data-" + u + "]", r);
            t&&!t.get("disabled") && t.getData(u) && (this.jumpTo(t.getData(u)), e.preventDefault())
        }, onResize:
        function() {
            var e = this.currentItem, t = this.maskEl.get("offsetWidth"), n = this.totalWidth;
            this.currentWidth = t, e !== 0 && (n <= t ? this.jumpTo(0, "none") : this.itemsPositions[e] + t >= n && this.jumpTo(this.itemsLength - 1, "none")), this.updateToolbar()
        }, swipe:
        function(e) {
            var t = 0, n = this.sliderEl, r = this.itemsLength, u = this.currentItem;
            switch (e.type) {
            case h:
                e.preventDefault();
                break;
            case l:
                this.loadDelayedImages();
                break;
            case f:
                e.detail.delta.absx > 30 && (t = e.detail.delta.x + this.itemsPositions[u], this.translate(n, - t, 0, i));
                break;
            case c:
                this.jumpTo(u);
                break;
            default:
                e.detail.direction === "left" ? this.jumpTo(o) : this.jumpTo(s)
            }
        }, updateIndicators:
        function() {
            var e = this.indicatorsEl;
            e && (e.removeClass(v), e.item(this.currentItem).addClass(v))
        }, updateToolbar:
        function() {
            var e = this.currentItem, t = this.currentWidth, n = this.totalWidth, r = this.prevEl, i = this.nextEl;
            r && i && (e + 1 === this.itemsLength || n <= t ? (i.setAttribute(N, N), i.addClass(w)) : (i.removeAttribute(N, N), i.removeClass(w)), e === 0 || n <= t ? (r.setAttribute(N, N), r.addClass(w)) : (r.removeAttribute(N, N), r.removeClass(w)))
        }
    }, {
        NS: "StencilGallery", CSS_SELECTED : d, CSS_ACTIVE : v, CSS_ITEM : m, CSS_SLIDER : g, CSS_INDICATORS : y, CSS_MASK : b, CSS_DISABLED : w, PREV_SELECTOR : E, NEXT_SELECTOR : S, requiresInit : !0
    }), e.Stencil.registerPlugin(k, "Gallery")
},
"@VERSION@",
{
    requires: ["stencil-base",
    "yui-base",
    "dom-base",
    "node-base",
    "node-style",
    "node-pluginhost",
    "transition"]
});
YUI.add("ape-location-lang-strings_en-au", function(e, t) {
    e.Intl.add("ape-location/strings", "en-AU", {
        DETECT_YOUR_LOCATION: "Detect your location",
        DETECT_YOUR_LOCATION_TITLE: "Detect your location",
        REMOVE_LOCATION: "Remove from favourite locations",
        REMOVE_LOCATION_TITLE: "Remove this location",
        ADD_LOCATION: "Add to favourite locations",
        ADD_LOCATION_TITLE: "Add this location",
        SAVE: "Save",
        SAVED: "Saved",
        VIEW_LOCATION_TITLE: "View your locations",
        LOCATION_PICKER: "Location picker",
        DETECT_MY_LOCATION: "Detect my location",
        ENTER_CITY_ZIPCODE_LABEL: "Enter City",
        ENTER_CITY_ZIPCODE_PLACEHOLDER: "Enter City",
        MANAGE_LOCATION: "Manage locations",
        GO: "Go",
        FAVORATE: "favourite",
        BROWSER_DENIED_PERMISSION: "Browser denied permission to get your current location",
        BROWSER_LOCATION_NOT_FOUND: "Browser was unable to find your current location",
        BROWSER_TIMEOUT_LOCATION: "Browser timed out while getting your location"
    })
}, "@VERSION@", {
    requires: ["intl"]
});
YUI.add("td-applet-astro-model", function(e) {
    "use strict";
    e.namespace("TD.Applet").AstroModel = e.Base.create("TDAppletAstroModel", e.Model, [e.Af.Sync], {
        resource: "astro",
        readonly: !0,
        consolidate: !1,
        idAttribute: "sign"
    }, {
        ATTRS: {
            sign: {
                value: null
            },
            date: {
                value: null
            },
            category: {
                value: null
            }
        }
    })
}, "@VERSION@", {
    requires: ["model", "af-sync"]
});
YUI.add("td-applet-astro-appletmodel", function(e) {
    "use strict";
    e.namespace("TD.Applet").HoroscopeAppletModel = e.Mjata.ModelBase.createModelClass("TDAppletHoroscopeAppletModel", "1.0", e.Af.AppletModel, [], {
        initializer: function() {
            this.after("settingsChange", e.bind(this.settingsChanged, this))
        },
        settingsChanged: function(e) {
            var t = this.getDataModel("astro");
            e.prevVal.sign !== e.newVal.sign && t.load({
                sign: e.newVal.sign
            })
        }
    })
}, "@VERSION@", {
    requires: ["mjata-model-base", "af-applet-model"]
});
YUI.add("td-applet-astro-listview", function(e) {
    "use strict";
    e.namespace("TD.Applet").AstroListView = e.Base.create("TDAppletAstroListView", e.Af.AppletView, [], {
        events: {
            "a.js-show-sign": {
                click: "selectSign"
            }
        },
        selectSign: function(e) {
            var t = this, n = this.get("model"), r = {
                sign: e.target.ancestor("[data-sign-id]").getData("sign-id"),
                category: n.getDataModel("astro").get("category")
            };
            e.preventDefault(), n.saveSettings(r, {}, function(e) {
                if (e) {
                    t.fireEvent("showmessage", {
                        level: "error",
                        token: "ERR_SAVE_SETTINGS"
                    });
                    return 
                }
            })
        }
    })
}, "@VERSION@", {
    requires: ["af-applet-view"]
});
YUI.add("td-applet-astro-signview", function(e) {
    "use strict";
    e.namespace("TD.Applet").AstroSignView = e.Base.create("TDAppletAstroSignView", e.Af.AppletView, [], {
        events: {
            "button.js-fetch-horoscope": {
                click: "dateArrows"
            },
            ".js-another-sign": {
                click: "showList"
            }
        },
        initializer: function() {
            var t = new Date, n = new Date, r = "", i = new Date, s = new Date;
            t.setDate(t.getDate() - 30), n.setDate(n.getDate() + 30), this.on("viewrender", function() {
                e.one("#prevBtn") && e.one("#nextBtn") && (r = e.one("#prevBtn").getAttribute("data-date"), i.setFullYear(r.substr(0, 4), r.substr(4, 2) - 1, r.substr(6, 2)), r = e.one("#nextBtn").getAttribute("data-date"), s.setFullYear(r.substr(0, 4), r.substr(4, 2) - 1, r.substr(6, 2)), i <= t && (e.one("#prevBtn").setAttribute("disabled", "1"), e.one("#prevBtn").addClass("Disabled")), s >= n && (e.one("#nextBtn").setAttribute("disabled", "1"), e.one("#nextBtn").addClass("Disabled")))
            })
        },
        dateArrows: function(e) {
            var t = {};
            t.date = e.target.ancestor("[data-date]") ? e.target.ancestor("[data-date]").getData("date") : e.target.getData("date"), this.refreshHoroscope(e, t)
        },
        refreshHoroscope: function(e, t) {
            var n = this;
            e.preventDefault(), t.sign || (t.sign = n.get("model").getDataModel("astro").get("sign")), t.category || (t.category = n.get("model").getDataModel("astro").get("category")), t.date || (t.date = n.get("model").getDataModel("astro").get("date")), n.get("model").getDataModel("astro").load(t, function() {
                n.fireEvent("showview", {
                    view: "main",
                    region: "main"
                })
            })
        },
        showList: function(e) {
            var t = this, n = this.get("model");
            n.getDataModel("astro").set("signName", ""), n.saveSettings({
                sign: "COS",
                category: "overview"
            }, {}, function(e) {
                if (e) {
                    t.fireEvent("showmessage", {
                        level: "error",
                        token: "ERR_SAVE_SETTINGS"
                    });
                    return 
                }
                t.fireEvent("showview", {
                    view: "list",
                    region: "main"
                })
            }), e.preventDefault()
        }
    })
}, "@VERSION@", {
    requires: ["af-applet-view"]
});
YUI.add("td-applet-astro-signheaderview", function(e) {
    "use strict";
    e.namespace("TD.Applet").AstroSignHeaderView = e.Base.create("TDAppletAstroSignHeaderView", e.Af.AppletView, [], {
        initializer: function() {
            var e = this.get("model"), t = this;
            e.after("settingsChange", function(n) {
                var r = e.getDataModel("astro"), i = n.newVal.sign, s = n.newVal.category;
                if (i === "COS")
                    return;
                s && r.set("category", n.newVal.category, {
                    silent: !1
                }), i && r.set("sign", n.newVal.sign, {
                    silent: !1
                }), t.refreshHoroscope(n, {
                    sign: i,
                    category: s
                })
            })
        },
        refreshHoroscope: function(e, t) {
            var n = this;
            e.preventDefault(), t.sign || (t.sign = n.get("model").getDataModel("astro").get("sign")), t.category || (t.category = n.get("model").getDataModel("astro").get("category")), t.date || (t.date = n.get("model").getDataModel("astro").get("date")), n.get("model").getDataModel("astro").load(t, function() {
                n.fireEvent("showview", {
                    view: "main",
                    region: "main"
                })
            })
        }
    })
}, "@VERSION@", {
    requires: ["af-applet-view"]
});
YUI.add("td-applet-astro-signfooterview", function(e) {
    "use strict";
    e.namespace("TD.Applet").AstroSignFooterView = e.Base.create("TDAppletAstroSignFooterView", e.Af.AppletView, [], {
        autoRender: !1,
        initializer: function() {
            this.get("model").after("change", e.bind(this.handleHoroscopeChange, this))
        },
        handleHoroscopeChange: function() {
            var e = this.get("model").getDataModel("astro").toJSON();
            e.signName === "Cosmic" || e.signName === "cosmic" || e.signName === "" ? e && e.landing_url ? this.get("container").one("a").set("href", e.landing_url) : this.get("container").one("a").set("href", "http://shine.yahoo.com/horoscope/") : e && e.link ? this.get("container").one("a").set("href", e.link) : this.get("container").one("a").set("href", "http://shine.yahoo.com/horoscope/")
        }
    })
}, "@VERSION@", {
    requires: ["af-applet-view", "mjata-model-store"]
});
YUI.add("td-applet-astro-templates-header", function(e, t) {
    dust.cache = dust.cache || {}, dust.cache[t] = function() {
        return function() {
            function e(e, n) {
                return e.write('    <div class="GridSpread">').write("\n        ").write('<h2 class="App-Title Grid-U">').helper("i18n_string", n, {}, {
                    _key: "TITLE"
                }).write("</h2>").write("\n       ").exists(n.getPath(!1, ["data", "astro", "signName"]), n, {
                    block: t
                }, null).write("\n    ").write("</div>").write("\n\n")
            }
            function t(e, t) {
                return e.write("\n           ").helper("select", t, {
                    block: n
                }, {
                    key: t.getPath(!1, ["data", "astro", "signName"])
                }).write("\n        ")
            }
            function n(e, t) {
                return e.write("\n                ").helper("eq", t, {
                    block: r
                }, {
                    value: "cosmic"
                }).write("\n                ").helper("eq", t, {
                    block: i
                }, {
                    value: "Cosmic"
                }).write("\n                ").helper("default", t, {
                    block: s
                }, null).write("\n            ")
            }
            function r(e, t) {
                return e
            }
            function i(e, t) {
                return e
            }
            function s(e, t) {
                return e.write("\n                    ").exists(t.getPath(!1, ["data", "astro", "multicategories"]), t, {
                    block: o
                }, null).write("\n                ")
            }
            function o(e, t) {
                return e.write("\n                        ").exists(t.getPath(!1, ["data", "astro", "notFallback"]), t, {
                    block: u
                }, null).write("\n                    ")
            }
            function u(e, t) {
                return e.write("\n                            ").write('<form class="Grid-U SelectBox astro-category" action="" data-sign-id="').reference(t.getPath(!1, ["data", "astro", "sign"]), t, "h").write('" data-date="">').write("\n                                ").write('<div class="SelectBox-Pick">').write("\n                                    ").write('<b class="SelectBox-Text">').write("\n                                        ").section(t.getPath(!1, ["data", "astro", "categories"]), t, {
                    block: a
                }, null).write("\n                                    ").write("</b>").write("\n                                    ").write('<i class="Icon">&#xe002;</i>').write("\n                                ").write("</div>").write("\n                                ").write('<select class="js-selectbox js-applet-action" data-plugin="selectbox" data-applet-action="savesettings" name="category" data-applet-actioncfg="src:form.astro-category;showview:main">').write("\n                                    ").section(t.getPath(!1, ["data", "astro", "categories"]), t, {
                    block: l
                }, null).write("\n                                ").write("</select>").write("\n                                ").write('<noscript><button type="submit">').helper("i18n_string", t, {}, {
                    _key: "SUBMIT"
                }).write("</button></noscript>").write("\n                            ").write("</form>").write("\n                        ")
            }
            function a(e, t) {
                return e.write("\n                                        ").helper("eq", t, {
                    block: f
                }, {
                    key: t.getPath(!1, ["data", "astro", "category"]),
                    value: t.get("category_id")
                }).write("\n                                        ")
            }
            function f(e, t) {
                return e.reference(t.get("category_text"), t, "h")
            }
            function l(e, t) {
                return e.write("\n                                    ").write("<option data-category=").reference(t.get("category_id"), t, "h").write(" value=").reference(t.get("category_id"), t, "h").write(" ").helper("eq", t, {
                    block: c
                }, {
                    key: t.getPath(!1, ["data", "astro", "category"]),
                    value: t.get("category_id")
                }).write(">").reference(t.get("category_text"), t, "h").write("</option>").write("\n                                    ")
            }
            function c(e, t) {
                return e.write(" selected")
            }
            return dust.register("td-applet-astro-templates-header", e), e
        }()
    }(), dust.cache["td-applet-astro:header.dust"] = dust.cache["td-applet-astro:header"] = dust.cache[t], e.Template._cache = e.Template._cache || {}, e.Template._cache["td-applet-astro/templates/header"] = function(e, n) {
        e = e || {}, dust.render(t, e, n)
    }
}, "@VERSION@", {
    requires: ["template-base", "dust"]
});
YUI.add("td-applet-astro-templates-fallback", function(e, t) {
    dust.cache = dust.cache || {}, dust.cache[t] = function() {
        return function() {
            function e(e, n) {
                return e.write('<ul class="Reset Grid W-100 Fz-s">').write("\n    ").section(n.getPath(!1, ["data", "astro", "signlist"]), n, {
                    block: t
                }, null).write("\n").write("</ul>").write("\n")
            }
            function t(e, t) {
                return e.write("\n    ").write('<li class="Grid-U-1-3 Mt-4" data-sign-id="').reference(t.get("sign_id"), t, "h").write('">').write("\n        ").write('<a class="js-show-sign Td-n rapidnofollow" href="').reference(t.getPath(!1, ["data", "astro", "landing_url"]), t, "h").reference(t.get("sign_name"), t, "h").write('">').write("\n            ").helper("img", t, {}, {
                    "class": "M-a D-b Mend-6 W-30 Fl-start",
                    src: n
                }).write("\n            ").write('<b class="D-b Ell Pstart-4">').helper("i18n_string", t, {}, {
                    _key: t.get("sign_id")
                }).write("</b>").write("\n            ").write('<b class="D-b Ell Pstart-4 Dimmed">').reference(t.get("date_range"), t, "h").write("</b>").write("\n        ").write("</a>").write("\n    ").write("</li>").write("\n    ")
            }
            function n(e, t) {
                return e.reference(t.getPath(!1, ["data", "astro", "images", t.get("sign_id")]), t, "h")
            }
            return dust.register("td-applet-astro-templates-fallback", e), e
        }()
    }(), dust.cache["td-applet-astro:fallback.dust"] = dust.cache["td-applet-astro:fallback"] = dust.cache[t], e.Template._cache = e.Template._cache || {}, e.Template._cache["td-applet-astro/templates/fallback"] = function(e, n) {
        e = e || {}, dust.render(t, e, n)
    }
}, "@VERSION@", {
    requires: ["template-base", "dust"]
});
YUI.add("td-applet-astro-templates-list", function(e, t) {
    dust.cache = dust.cache || {}, dust.cache[t] = function() {
        return function() {
            function e(e, n) {
                return e.write('<ul class="Reset Grid W-100 Fz-s">').write("\n    ").section(n.getPath(!1, ["data", "astro", "signlist"]), n, {
                    block: t
                }, null).write("\n").write("</ul>").write("\n")
            }
            function t(e, t) {
                return e.write("\n    ").write('<li class="Grid-U-1-3 Mt-4" data-sign-id="').reference(t.get("sign_id"), t, "h").write('">').write("\n        ").write('<a class="js-show-sign Td-n rapidnofollow" href="#">').write("\n            ").helper("img", t, {}, {
                    "class": "M-a D-b Mend-6 W-30 Fl-start",
                    src: n
                }).write("\n            ").write('<b class="D-b Ell Pstart-4">').helper("i18n_string", t, {}, {
                    _key: t.get("sign_id")
                }).write("</b>").write("\n            ").write('<b class="D-b Ell Pstart-4 Dimmed">').reference(t.get("date_range"), t, "h").write("</b>").write("\n        ").write("</a>").write("\n    ").write("</li>").write("\n    ")
            }
            function n(e, t) {
                return e.reference(t.getPath(!1, ["data", "astro", "images", t.get("sign_id")]), t, "h")
            }
            return dust.register("td-applet-astro-templates-list", e), e
        }()
    }(), dust.cache["td-applet-astro:list.dust"] = dust.cache["td-applet-astro:list"] = dust.cache[t], e.Template._cache = e.Template._cache || {}, e.Template._cache["td-applet-astro/templates/list"] = function(e, n) {
        e = e || {}, dust.render(t, e, n)
    }
}, "@VERSION@", {
    requires: ["template-base", "dust"]
});

