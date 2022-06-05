define('dust!site/deckInspectah/DeckInspectahView.dust', ['dust-helpers', 'dust-core'], function(helpers) {
    (function() {
        dust.register("site/deckInspectah/DeckInspectahView.dust", body_0);
        function body_0(chk, ctx) {
            return chk.write("<div class=\"w-deck-inspectah-overlay w-deck-inspectah-hidden w-deck-inspectah-loading w-deck-inspectah-image-loading ").exists(ctx.get(["sidebarEnabled"], false), ctx, {
                "block": body_1
            }, null).write("\"><div class=\"w-deck-inspectah-image-container\"><div id=\"w-deck-inspectah-image-container-inner\" class=\"w-deck-inspectah-image-container-inner\"></div><div class=\"w-deck-inspectah-spinner-container\"></div></div><a href=\"#close\" class=\"w-deck-inspectah-control w-deck-inspectah-close disabled\"><div class=\"icon\"/></a><a href=\"#previous\" class=\"w-deck-inspectah-control w-deck-inspectah-switch-image previous disabled\"><div class=\"icon\"/></a><a href=\"#next\" class=\"w-deck-inspectah-control w-deck-inspectah-switch-image next disabled\"><div class=\"icon\"/></a><div class=\"w-deck-inspectah-sidebar\"><div class=\"w-deck-inspectah-sidebar-inner\"></div></div></div>");
        }
        function body_1(chk, ctx) {
            return chk.write("sidebar-active");
        }
        return body_0;
    })();
    return function(data, callback) {
        dust.render('site/deckInspectah/DeckInspectahView.dust', helpers.push(data), callback);
    }
});
