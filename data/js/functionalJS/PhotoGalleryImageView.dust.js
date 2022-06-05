define('dust!site/modules/photo_gallery/PhotoGalleryImageView.dust', ['dust-helpers', 'dust-core'], function(helpers) {
    (function() {
        dust.register("site/modules/photo_gallery/PhotoGalleryImageView.dust", body_0);
        function body_0(chk, ctx) {
            return chk.write("<div class=\"webs-image-wrapper-1\"><div class=\"webs-image-crop\"").exists(ctx.get(["height"], false), ctx, {
                "block": body_1
            }, null).write("><img src=\"").reference(ctx.get(["url"], false), ctx, "h").write("\" alt=\"").reference(ctx.get(["title"], false), ctx, "h").write("\" style=\"top: ").reference(ctx.get(["top"], false), ctx, "h").write("px;left: ").reference(ctx.get(["left"], false), ctx, "h").write("px;width: ").reference(ctx.get(["imageWidth"], false), ctx, "h").write("%;\"></div></div><div class=\"image-crop-explanation\"><div class=\"explanation-circle\"><div class=\"explanation-container\"><div class=\"explanation-image\"></div><div class=\"explanation-text\">").section(ctx.get(["l"], false), ctx, {
                "block": body_2
            }, null).write("</div></div></div></div>");
        }
        function body_1(chk, ctx) {
            return chk.write("style=\"padding-bottom: ").reference(ctx.get(["height"], false), ctx, "h").write("px\"");
        }
        function body_2(chk, ctx) {
            return chk.write("webs.module.photo_gallery.imageCropExplanation");
        }
        return body_0;
    })();
    return function(data, callback) {
        dust.render('site/modules/photo_gallery/PhotoGalleryImageView.dust', helpers.push(data), callback);
    }
});
