define([
	"jquery",
	"backbone"
], function($, Backbone){
	return Backbone.View.extend({
		tagName: 'div',
		className: 'spinner-circle-container',

		render: function(){
			this.$el.html($('<div class="image-loading-spinner">&nbsp;</div><div class="css-spinner-circle"></div>'));
			return this;
		}
	});
});
