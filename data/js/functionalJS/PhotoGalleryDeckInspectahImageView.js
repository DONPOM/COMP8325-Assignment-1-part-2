define([
	'backbone',
	'underscore',
	'dust!site/modules/photo_gallery/PhotoGalleryDeckInspectahImageView.dust'
], function (Backbone, _, template) {

	var PhotoGalleryDeckInspectahImageView = Backbone.View.extend({

		initialize: function () {
			this._bind();
		},

		changeModel: function(newModel) {
			this._unbind();
			this.model = newModel;
			this._bind();
			this.render();
		},

		render: function () {
			var self = this;

			template(this.model.attributes, function (err, html) {
				self.$el.html(html);
			});
		},

		// private
		_bind : function() {
			this.model.on('change', this.render, this);
		},

		_unbind : function() {
			this.model.off(null, null, this);
		}
	});

	return PhotoGalleryDeckInspectahImageView;
});