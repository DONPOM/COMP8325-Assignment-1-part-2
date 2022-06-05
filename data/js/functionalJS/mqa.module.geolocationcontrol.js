MQA.Loader.registerCss("geolocationcontrol", ".geoloc{position:absolute;width:31px;height:32px;background-repeat:no-repeat;background-position:-269px -240px;cursor:pointer;-moz-user-select:none;-webkit-user-select:none;}.geoloc.active{background-position:-269px -273px;}.geoloc.sat{background-position:-269px -461px;}.geoloc.sat.active{background-position:-269px -496px;}.geoloc-loader{position:absolute;top:50%;left:50%;margin:-8px 0 0 -8px;width:16px;height:16px;}.geoloc .geoloc-loader{display:none;}.geoloc.loading .geoloc-loader{display:inline;}.geoloc.active.loading .geoloc-icon-light,.geoloc.sat.active.loading .geoloc-icon-light{display:none;}.geoloc-icon{position:absolute;top:50%;left:50%;margin:-14px 0 0 -12.5px;}.geoloc .geoloc-icon-dark{display:none;}.geoloc.sat .geoloc-icon-light{display:none;}.geoloc.sat .geoloc-icon-dark{display:inline;}.geoloc.sat.active .geoloc-icon-light{display:inline;}.geoloc.sat.active .geoloc-icon-dark{display:none;}");
MQA.withModule("controlbase", "geolocation", function() {
    var H = window.MQA, I = H.Loader.resourcePath, K = H.EventUtil, J = K.EventCallback, A = K.observe, G = H.EventManager.clearAllListeners, L = H.Util, N = L.addClass, M = L.removeClass;
    H.Loader.requireCss("geolocationcontrol");
    var C = "geolocation", D = ["div#root.geoloc", ["img#icon_light.geoloc-icon geoloc-icon-light", ["s", ""], "img#icon_dark.geoloc-icon geoloc-icon-dark", ["s", ""], "img#loader.geoloc geoloc-loader", ["s", ""]]];
    function F(P, O) {
        return (O / P) * 2
    }
    function E(O, S, T, U) {
        var X = O.getZoomLevel(), V = O.getSize(), W = F(O.getResolution(), T), P, Q, R;
        if (U) {
            while (W < 25 && T > 10 && X <= H.MAXZOOM) {
                X = X + 1;
                W = F(H._RESOLUTIONS[X], T)
            }
            while ((V.width < W || V.height < W) && X >= 1) {
                X = X - 1;
                W = F(H._RESOLUTIONS[X], T)
            }
        }
        P =- W / 2;
        Q = H.Graphics.createSurface(W + 15, W + 15);
        R = Q.ellipse();
        R.setPoints([{
            x: 0,
            y: 0
        }, {
            x: W,
            y: W
        }
        ]);
        R.setColor("#667F33");
        R.setColorAlpha("0.5");
        R.setFillColor("#99cc66");
        R.setFillColorAlpha("0.25");
        R.add();
        A(Q.elt, "mousedown", function(Y) {
            O._onDOMEvent(Y)
        });
        S.setContent("info", Q.elt, P, P - 10, true, "shape");
        return X
    }
    function B() {
        this.id = "geolocation";
        this.active = false
    }
    B.prototype = new H.Control();
    H.extend(B.prototype, {
        initialize: function(R) {
            var P = this, Q = P.html = H.Util.html(D), O = Q.root;
            P.map = R;
            P.elem = O;
            O.style.backgroundImage = "url(" + I("images/sprite_map_controls.png") + ")";
            O.title = "Find Me";
            if (R.mapType === "hyb" || R.mapType === "sat") {
                Q.icon_dark.src = I("images/find_me_sat.png");
                N(O, "sat")
            } else {
                Q.icon_light.src = I("images/find_me_map.png")
            }
            if (!H.Graphics || H.Geolocation.stub) {
                O.style.display = "none";
                return 
            }
            A(O, "click", function(S) {
                P.toggle()
            });
            A(O, "touchstart", function(S) {
                P.toggle()
            });
            A(O, "mouseover", function(S) {
                if (P.active) {
                    return 
                }
                N(O, "active")
            });
            A(O, "mouseout", function(S) {
                if (P.active) {
                    return 
                }
                M(O, "active")
            });
            H.EventManager.addListener(P.map, "maptypechanged", P.setMapType, P)
        },
        activate: function() {
            var P = this, R = P.map, Q = P.html, O = Q.root;
            P.active = true;
            if (!Q.loader.src) {
                Q.loader.src = I("images/geolocation_loader.gif")
            }
            N(O, "active");
            N(O, "loading");
            $a("MQ10GeolocationActivate");
            H.Geolocation.getCurrentPosition(function(Z) {
                var U = Z.coords, V = U.accuracy, Y = {
                    lat: U.latitude,
                    lng: U.longitude
                }, X = R.getShapeCollection(C), W, T, S;
                M(O, "loading");
                if (!X) {
                    X = new H.ShapeCollection();
                    X.setName(C);
                    R.addShapeCollection(X)
                }
                X.removeAll();
                T = new H.Icon(I("images/find_me.png"), 14, 20);
                W = new H.Poi(Y, T);
                W.setIconOffset({
                    x: - 7,
                    y: - 20
                });
                W.setShadow(null);
                W.accuracy = V;
                S = E(R, W, V, true);
                X.add(W);
                P.map.setCenter(Y, S);
                P.onLocate(W, Z)
            }, function(S) {
                P.deactivate()
            }, {
                enableHighAccuracy: true,
                maximumAge: 60000,
                timeout: 10000
            })
        },
        deactivate: function() {
            var P = this, Q = P.map, R = Q.getShapeCollection(C), O = P.html.root;
            P.active = false;
            M(O, "active");
            M(O, "loading");
            if (R) {
                R.removeAll()
            }
            $a("MQ10GeolocationDeactivate")
        },
        onLocate: function(O, P) {},
        setMapType: function(P) {
            var Q = this.html, O = Q.root;
            switch (P.mapType) {
            case H.MAP_TYPE.SAT:
            case H.MAP_TYPE.HYB:
                if (!Q.icon_dark.src) {
                    Q.icon_dark.src = I("images/find_me_sat.png")
                }
                N(O, "sat");
                break;
            default:
                if (!Q.icon_light.src) {
                    Q.icon_light.src = I("images/find_me_map.png")
                }
                M(O, "sat");
                break
            }
        },
        setZoom: function(P) {
            var O = this, R = O.map, T = R.getShapeCollection(C), Q, S;
            if (!T || T.getSize() === 0) {
                return 
            }
            S = T.getAt(0);
            Q = S.accuracy || 50;
            E(R, S, Q, false)
        },
        toggle: function() {
            this[this.active ? "deactivate": "activate"]()
        },
        dispose: function() {
            var P = this, O = P.html.root;
            P.deactivate();
            G(P);
            O.parentNode.removeChild(O)
        }
    });
    H.GeolocationControl = B;
    H.Loader._moduleLoaded("geolocationcontrol")
});
