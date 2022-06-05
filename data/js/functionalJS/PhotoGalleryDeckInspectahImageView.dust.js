define('dust!site/modules/photo_gallery/PhotoGalleryDeckInspectahImageView.dust', ['dust-helpers', 'dust-core'], function(helpers) {
    (function() {
        dust.register("site/modules/photo_gallery/PhotoGalleryDeckInspectahImageView.dust", body_0);
        function body_0(chk, ctx) {
            return chk.write("<img src=\"").reference(ctx.get(["url"], false), ctx, "h").write("\" id=\"deckInspectahImage\"/>");
        }
        return body_0;
    })();
    return function(data, callback) {
        dust.render('site/modules/photo_gallery/PhotoGalleryDeckInspectahImageView.dust', helpers.push(data), callback);
    }
});
