(function() {
    var A, D = MQA.mixin, B = MQA.Util.subClass, J = MQA.EventUtil, E = MQA.EventManager, I = J.stop, H = J.isLeftClick, C = E.trigger, K = MQA.Log.debug;
    function G(M, L) {
        this.stateName = M;
        this.cursor = "pointer";
        if (L) {}
    }
    G.prototype = {
        isValid: function() {
            return true
        },
        activate: function() {
            var L = this.poi;
            if (this.icon) {
                L.setContentVisible("-icon", true)
            }
            this.activated = true;
            if (this.icon) {
                if (!this.inited) {
                    this.applyIcon();
                    this.inited = true
                }
            }
        },
        deactivate: function() {
            var L = this.poi;
            L.setContentVisible("-icon", false);
            this.activated = false
        },
        getIconOffset: function() {
            var M = this.iconOffset, L = this.icon;
            if (!M) {
                M = L ? {
                    x: - parseInt(L.width / 2),
                    y: - parseInt(L.height / 2)
                } : {
                    x: 0,
                    y: 0
                }
            }
            return M
        },
        setIconOffset: function(L) {
            this.iconOffset = L;
            this.applyIcon(true)
        },
        getIcon: function() {
            return this.icon
        },
        setIcon: function(L) {
            this.icon = L;
            this.applyIcon()
        },
        applyIcon: function(L) {
            var P = this.poi, N = this.icon, O = this.getIconOffset() || {
                x: 0,
                y: 0
            }, M;
            if (P) {
                if (L) {
                    P.setContentOffset(this.stateName + "-icon", O.x, O.y)
                } else {
                    M = N && N.createElement();
                    if (M) {
                        M.style.cursor = this.cursor
                    }
                    P.setContent(this.stateName + "-icon", M, O.x, O.y, "icon", P.zIndex);
                    P.setContentVisible(this.stateName + "-icon", this.activated)
                }
            }
        },
        getBounds: function(L, R) {
            var Q = this.poi, O = this.getIconOffset(), M = this.icon, N;
            N = {
                x: L,
                y: R,
                anchorX: L,
                anchorY: R,
                width: 0,
                height: 0,
                icon: {
                    x: O.x,
                    y: O.y,
                    width: (M && M.width) || 0,
                    height: (M && M.height) || 0
                }
            };
            function P(S) {
                S.offsetX = S.x;
                S.offsetY = S.y;
                S.x += L;
                S.y += R;
                var U = S.x + S.width, T = S.y + S.height;
                lrX = N.x + N.width, lrY = N.y + N.height;
                if (N.x > S.x) {
                    N.x = S.x
                }
                if (N.y > S.y) {
                    N.y = S.y
                }
                if (lrX < U) {
                    N.width += (U - lrX)
                }
                if (lrY < T) {
                    N.height += (T - lrY)
                }
            }
            P(N.icon);
            return N
        }
    };
    A = function(L, M) {
        MQA.BasePoi.call(this);
        this.zIndex = "hover_poi";
        this.xmlUrl = "";
        this.xmlUrlVideo = "";
        this._dspStates = {
            "": new G("", true)
        };
        this.stateStack = [""];
        if (L) {
            this.setLatLng(L)
        }
        if (M) {
            this.setIcon(M)
        }
        this.draggable = false;
        this.addDOMEvent("click", "mouseover", "mouseout", "mousedown")
    };
    var F = B(MQA.BasePoi, {
        _onDOMEvent: function(M) {
            if (M.type == "mousedown") {
                this._isDown = true;
                I(M);
                return 
            } else {
                if (M.type == "mouseup" && this._isDown) {
                    this._isDown = false;
                    I(M);
                    return 
                }
            }
            N = this["_onDOM" + M.type];
            if (N) {
                N.call(this, M)
            } else {
                var L = new MQA.Event("MQA.HoverPoi." + M.type, this), N;
                L.button = H(M) ? MQA.BUTTON_MQ_LEFT : MQA.BUTTON_MQ_RIGHT;
                L.domEvent = M;
                C(this, M.type, L)
            }
        },
        _onDOMclick: function(M) {
            var L = new MQA.Event("MQA.HoverPoi.click"), N;
            L.button = H(M) ? MQA.BUTTON_MQ_LEFT : MQA.BUTTON_MQ_RIGHT;
            L.domEvent = M;
            L.srcObject = this;
            L.xmlUrl = this.xmlUrl;
            L.xmlUrlVideo = this.xmlUrlVideo;
            C(this, "click", L)
        },
        layerInit$After: function() {
            this._activated = true;
            this.setState(this.state);
            this.setBias(this.bias)
        },
        getDisplayState: function(M) {
            var L = this._dspStates[M || ""];
            if (!L) {
                L = this._dspStates[M || ""] = new G(M)
            }
            return L
        },
        setIconOffset: function(M, L) {
            this.getDisplayState(L).setIconOffset(M)
        },
        getIconOffset: function(L) {
            return this.getDisplayState(L).iconOffset || {
                x: 0,
                y: 0
            }
        },
        setIcon: function(L, M) {
            this.getDisplayState(M).setIcon(L)
        },
        getIcon: function(L) {
            return this.getDisplayState(L).icon
        },
        getCursor: function() {
            return this.cursor
        },
        setCursor: function(N) {
            this.cursor = N;
            var M = this.poi, L;
            if (M) {
                L = M.getContent("-icon");
                if (L) {
                    L.style.cursor = N
                }
            }
        },
        setState: function(L) {
            this.stateStack = [L];
            return this._applyState(L)
        },
        _applyState: function(M) {
            if (this._activated) {
                var L = this._dspStates[M || ""], N = this._curDspState;
                if (L && L.isValid()) {
                    if (N) {
                        N.deactivate()
                    }
                    L.poi = this;
                    L.activate();
                    this._curDspState = L;
                    return true
                } else {
                    return false
                }
            } else {
                return false
            }
        },
        getState: function() {
            return this.stateStack[this.stateStack.length - 1] || ""
        },
        getCurrentBounds: function() {
            var M = this._elt, O = this._dspStates[this.getState()], L = this.bias, N = O.getBounds(this._pxX, this._pxY);
            N.biasX = (L && L.x) || 0;
            N.biasY = (L && L.y) || 0;
            return N
        },
        _applyZIndex: function() {
            var L = this.zIndex, M = this;
            this._cnts.each(function(N, O) {
                if (O.match(/\-icon/)) {
                    M.setContentZIndex(O, L)
                }
            })
        }
    });
    A.prototype = F;
    F.defineProperty("infoTitleHTML");
    F.defineProperty("infoContentHTML");
    F.defineProperty("rolloverContent");
    F.defineProperty("zIndex", null, "poi", "_applyZIndex");
    D(A.prototype, {
        dispose$Before: function() {
            if (!this.map) {
                return 
            }
            var L = this.map.windowManager;
            L.close("rolloverwindow", this)
        },
        onEvent$After: function(L, N) {
            var M = this.map.windowManager;
            switch (L) {
            case"mouseover":
                if (this._isMouseOver) {
                    return 
                }
                this._isMouseOver = true;
                M.onPoiMouseOver(this, true);
                I(N);
                break;
            case"mouseout":
                if (this._isMouseOver) {
                    this._isMouseOver = false;
                    M.onPoiMouseOver(this, false)
                }
                I(N);
                break
            }
        },
        onEvent$Before: function(L, N) {
            var M = this.map.windowManager;
            switch (L) {
            case"click":
                I(N);
                break
            }
        },
        onWindowClose: function(L) {
            if (L === this.rolloverWindow) {
                delete this.rolloverWindow;
                this._isRollover = 0
            }
        }
    });
    MQA.HoverPoi = A;
    MQA.Loader._moduleLoaded("hoverpoi")
})();
