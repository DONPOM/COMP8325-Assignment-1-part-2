(function(a) {
    function h() {
        try {
            return "1" === a.cookie("mobile")
        } catch (h) {
            return !1
        }
    }
    a.modules.shoppingcart = function() {
        function t(b, n) {
            function p() {
                e.slideUp(d, function() {
                    a(this).remove()
                })
            }
            var e, d = 200;
            "success" === b.status && (c.html(b.payload.cart), !a.jimdoData.isDevkitTemplateUsed || h() ? (h() && f.show().find("span").html(b.payload.totalCount), c.find(".cc-m-input-numeric").numericInput({
                change: function() {
                    g.spinnerchange.apply(this, arguments)
                }
            })) : (a(".cc-shop-addtocard-response").remove(), e = a(b.payload.addToCartResponse).hide(),
            a("body").prepend(e), e.slideDown(d), e.actionController({
                _closeClick: p
            }, {
                events: "change click"
            }), window.setTimeout(p, 1E4)), g.loading(!1));
            a.isFunction(n) && n(b)
        }
        function d() {
            var b = 0;
            a.each(c.find(".cc-sidecart-item input"), function() {
                b += parseInt(this.value, 10)
            });
            return b
        }
        var c = a("#cc-sidecart-wrapper"), f, g = a.factory("SaleUtils", [{
            $container: c,
            type: "sidecart"
        }
        ]);
        c.find(".cc-m-input-numeric").numericInput({
            change: function() {
                g.spinnerchange.apply(this, arguments)
            }
        });
        c.bind("shoppingcartinsert", function(b,
        c, d) {
            g.loading(!0);
            a.sync({
                url: g.controller + "insert",
                type: "post",
                data: c,
                success: function(a) {
                    t(a, d)
                }
            })
        });
        a("body").browserMessage({
            tests: {
                shop_maintenance: {
                    test: function() {
                        return a.jimdoData.shop && a.jimdoData.shop.maintenance
                    },
                    message: a.jimdoData.tr.shop.maintenance
                }
            },
            autoRun: !/^(www|\w{2}).jimdo.com/.test(window.location.hostname)
        });
        if (h()) {
            var k = a(window), q = a("#cc-m-container"), r = navigator.userAgent.match(/iphone/i);
            f = a('\x3cdiv id\x3d"cc-sidecart-bar"\x3e\x3ca href\x3d"javascript:;"\x3e' + a.jimdoData.tr.shop.cartBar +
            " (\x3cspan\x3e" + d() + "\x3c/span\x3e)\x3c/a\x3e\x3c/div\x3e").css({
                position: r ? "absolute": "fixed"
            }).appendTo("body").click(function() {
                var b = c.offset().top - s - parseInt(q.css("marginTop"), 10);
                a("html, body").animate({
                    scrollTop: b
                }, l)
            });
            var s = f.height(), l = function() {
                q.css({
                    marginTop: 0 === k.scrollTop() && 0 < d() ? s: 0
                });
                r && f.css({
                    top: k.scrollTop()
                })
            };
            l();
            var m;
            k.scroll(function() {
                m && m();
                m = a.defer(l, 30)
            });
            0 < d() && f.show()
        }
    }
})(jimdoGen002);
