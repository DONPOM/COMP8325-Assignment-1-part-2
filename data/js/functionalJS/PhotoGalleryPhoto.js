define(["backbone"], function (Backbone) {
	return Backbone.Model.extend({
		idAttribute: "url",

		defaults: {
			url: '',
			title: '',
			caption: '',
			top: 0,
			left: 0,
			imageWidth: 100,
			_i_: 0,
			imageType: 'webs',
			webs: {
				path: ''
			}
		}
	});
});
