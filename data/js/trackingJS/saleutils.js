(function(k) {
    k.reg("SaleUtils", function(c, g) {
        var h = this, l = {
            loadingClass: "cc-cart-loading",
            controller: "/app/shop/shoppingcart/"
        }, a;
        this.init = function() {
            a = c.extend({}, l, g);
            this.controller = a.controller
        };
        this.loading = function(c) {
            a.$container[c ? "addClass": "removeClass"](a.loadingClass);
            return this
        };
        this.spinnerchange = function(g) {
            var f = c(this), d = {};
            d.newcount = g;
            d.count = f.data("product-count");
            d.item_id = f.data("product-item-id");
            d.hash = f.data("product-hash");
            h.loading(!0);
            c.sync({
                url: a.controller + "updatecount",
                type: "post",
                data: d,
                success: function(b) {
                    h.loading(!1);
                    if ("success" === b.status) {
                        if ("sidecart" === a.type) {
                            a.$container.html(b.payload.cart);
                            var e;
                            try {
                                e = "1" === k.cookie("mobile")
                            } catch (d) {
                                e=!1
                            }
                            e && (e = c("#cc-sidecart-bar"), 0 === b.payload.totalCount ? e.hide() : e.find("span").html(b.payload.totalCount))
                        } else 
                            a.$container.html(b.payload.checkout);
                        a.$container.find(".cc-m-input-numeric").numericInput({
                            change: function() {
                                h.spinnerchange.apply(this, arguments)
                            }
                        })
                    } else 
                        b.payload.reload && (b.onClose = function() {
                            location.reload()
                        }),
                        f.val(b.payload.count).message(b)
                }
            })
        }
    })
})(jimdoGen002);
