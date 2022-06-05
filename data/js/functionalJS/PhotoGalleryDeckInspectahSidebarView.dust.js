define('dust!site/modules/photo_gallery/PhotoGalleryDeckInspectahSidebarView.dust', ['dust-helpers', 'dust-core'], function(helpers) {
    (function() {
        dust.register("site/modules/photo_gallery/PhotoGalleryDeckInspectahSidebarView.dust", body_0);
        function body_0(chk, ctx) {
            return chk.exists(ctx.get(["titleEnabled"], false), ctx, {
                "block": body_1
            }, null).exists(ctx.get(["captionEnabled"], false), ctx, {
                "block": body_2
            }, null).exists(ctx.get(["shareEnabled"], false), ctx, {
                "block": body_3
            }, null);
        }
        function body_1(chk, ctx) {
            return chk.write("<div class=\"w-deck-inspectah-sidebar-title w-font-lato\">").reference(ctx.get(["title"], false), ctx, "h").write("</div>");
        }
        function body_2(chk, ctx) {
            return chk.write("<div class=\"w-deck-inspectah-sidebar-caption w-font-lato\">").reference(ctx.get(["caption"], false), ctx, "h").write("</div>");
        }
        function body_3(chk, ctx) {
            return chk.write("<div class=\"w-deck-inspectah-sidebar-share\"><!-- Not currently active --></div>");
        }
        return body_0;
    })();
    return function(data, callback) {
        dust.render('site/modules/photo_gallery/PhotoGalleryDeckInspectahSidebarView.dust', helpers.push(data), callback);
    }
});
