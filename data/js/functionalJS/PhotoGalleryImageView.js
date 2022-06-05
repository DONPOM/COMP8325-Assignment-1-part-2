define([
	'backbone',
	'underscore',
	'translate!webs.module.photo_gallery',
	'dust!site/modules/photo_gallery/PhotoGalleryImageView.dust',
	'ui/WebsSpinnerView'
], function (Backbone, _, translate, template, WebsSpinnerView) {

	var PhotoGalleryImageView = Backbone.View.extend({

		className: 'image-edit-wrapper',

		initialize: function () {
			this.imageReady = !!this.options.imageReady;
			this._bind();
			this.spinner = new WebsSpinnerView();
		},

		render: function () {
			var self = this;

			if(this.imageReady) {
				template(this.model.attributes, function (err, html) {
					self.$el.html(html);
				});
			}
			else {
				var placeholder = $('<div class="webs-image-placeholder-wrap"><div class="webs-image-placeholder"></div></div>');
				placeholder.find('.webs-image-placeholder').append(this.spinner.render().el);
				self.$el.html(placeholder);
			}
		},

		changeModel: function(newModel, imageReady) {
			this._unbind();
			this.model = newModel;
			this._bind();
			this._renderNewModel(imageReady);
		},

		removeLoadingCover: function() {
			this._renderNewModel(true);
		},

		// private
		_bind : function() {
			this.model.on('change', this.render, this);
		},

		_unbind : function() {
			this.model.off(null, null, this);
		},

		_renderNewModel: function(hideCover) {
			this.imageReady = !!hideCover;
			this.render();
		}
	});

	return PhotoGalleryImageView;
});