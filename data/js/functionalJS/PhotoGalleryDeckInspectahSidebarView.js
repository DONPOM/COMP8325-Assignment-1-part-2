define([
	'backbone',
	'underscore',
		'dust!site/modules/photo_gallery/PhotoGalleryDeckInspectahSidebarView.dust'
], function (Backbone, _, template) {

	var PhotoGalleryDeckInspectahSidebarView = Backbone.View.extend({
		initialize: function () {
			this._bind();

			this.titleEnabled = !!this.options.titleEnabled;
			this.captionEnabled = !!this.options.captionEnabled;
			this.socialEnabled = !!this.options.socialEnabled;
		},

		changeModel: function(newModel) {
			this._unbind();
			this.model = newModel;
			this._bind();
			this.render();
		},

		render: function () {
			var self = this;

			template(_.extend({
					titleEnabled: this.titleEnabled,
					captionEnabled: this.captionEnabled,
					socialEnabled: this.socialEnabled
				}, this.model.attributes), function (err, html) {
				self.$el.html(html);
			});
		},

		// private
		_bind: function() {
			this.model.on('change', this.render, this);
		},

		_unbind: function() {
			this.model.off(null, null, this);
		}
	});

	return PhotoGalleryDeckInspectahSidebarView;
});