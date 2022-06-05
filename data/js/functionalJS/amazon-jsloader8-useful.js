
(function ($Nav) {
"use strict";


    $Nav.when('$', 'data', 'flyout.yourAccount')
        .run("BuyitAgain-Render-Black-Belt", 
        function ($, data, yaFlyout) {

            var renderBuyItAgain = function (biaData) {
                if (biaData.status) {
                    navbar.sidePanel({
                        flyoutName: 'yourAccount',
                        data: {html: biaData.faceouts.join('')}
                    });
                }
            };

            yaFlyout.sidePanel.onData(function() {
                enableInlineAddToCart($); 
            }); 

            yaFlyout.onRender(function() {
                $.ajax({
                    url: '/gp/bia/external/bia-hcb-ajax-handler.html',
                    data: {"biaHcbRid":"1BB039VX1XJ9CTCMZHE6"},
                    dataType: 'json', 
                    timeout: 4*1000,

                    success: function (data) { 
                        renderBuyItAgain(data); 
                    },

                    error: function (jqXHR, textStatus, errorThrown) {
                    }
                });
            });


    var updateNavCartQty = function(qty) {
        if (typeof window.navbar === 'object' && typeof window.navbar.setCartCount === 'function') {
            window.navbar.setCartCount(qty);
        }
    };

    var addToCart = function(params, callback) {
        $.ajax({
           url: '/gp/bia/external/bia-cart-ajax-handler.html',
           data: params,
           dataType: 'json', 
           timeout: 2000,
           success: function(response) { callback(response); },
           error: function() { callback({ok:0}); }
        });
    };

    var enableInlineAddToCart = function ($) {
        if ($(".bia-inline-cart-form").length === 0) {
            return;
        }

        var inlineAddToCartHandler = function(e) {
            e.preventDefault();

            var $target = $(e.target);
            var $item = $target.parents(".bia-item");
            var $submit = $item.find(".bia-cart-submit");
            var params = $target.attr('data-order');

            $submit.attr("disabled", true);
            $item.find(".bia-action-button").addClass("bia-action-button-disabled");

            addToCart(params, 
                function(response) {
                    if(response && response.ok && response.ok === '1') {
                        $item.find(".bia-faceout").hide();
                        $item.find(".bia-cart-action").show();
                        updateNavCartQty(response.numActiveItemsInCart); 
                        //TODO: add metric
                    } else {
                        $target.unbind("submit", inlineAddToCartHandler);
                        $submit.attr("disabled", false);
                        $submit.click();
                        //TODO: add metric
                    }
                }
            );
        };

        $(".bia-inline-cart-form").bind("submit", inlineAddToCartHandler);
    };

        });

})(window.$Nav);
