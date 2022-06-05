define([
	'jquery',
	'backbone',
	'underscore',
	'ui/WebsSpinnerView',
	'dust!site/deckInspectah/DeckInspectahView.dust'
], function ($, Backbone, _, WebsSpinnerView, template) {

	var DeckInspectahView = Backbone.View.extend({
		// view properties
		selectedImage : 0,
		sidebarEnabled : false,
		imageView: null,
		sidebarView: null,

		events: {
			'click .previous' : 'previousImage',
			'click .next' : 'nextImage',
			'click .w-deck-inspectah-close' : 'close'
		},

		initialize: function () {
			this.model.on('change', this.updateImage, this);

			this.selectedImage = this.options.selectedImage || this.selectedImage;
			this.sidebarEnabled = this.options.titleEnabled || this.options.captionEnabled || this.options.socialEnabled;

			this.imageView =  new this.options.imageViewClass({model: this._getSelectedModel() });
			this.sidebarView = new this.options.sidebarViewClass({
				model: this._getSelectedModel(),
				titleEnabled : this.options.titleEnabled,
				captionEnabled : this.options.captionEnabled,
				socialEnabled : this.options.socialEnabled
			});

			this.spinner = new WebsSpinnerView();
		},

		render: function () {
			var self = this;

			template(_.extend({ sidebarEnabled : this.sidebarEnabled }, self._getSelectedModel().attributes), function (err, html) {
				self.$el.html(html);
				self.$('.w-deck-inspectah-spinner-container').html(self.spinner.render().el);
				self.$('.w-deck-inspectah-loading').removeClass('w-deck-inspectah-loading');
				self.updateImage();
			});
		},

		show: function() {
			this.render();
			this.$el.show(); // make sure the jquery object isn't hidden

			this.$('.w-deck-inspectah-hidden').removeClass('w-deck-inspectah-hidden'); // start the animation

			// special case - handle body-level keypress events
			$(document).on('keydown.deckinspectah', $.proxy(this.keydownHandler, this));
		},

		updateImage: function() {
			// add the loading class to start transitions
			this.$('.w-deck-inspectah-overlay').addClass('w-deck-inspectah-image-loading');

			// grab the old image (if it exists)
			var curImg = this.$('.w-deck-inspectah-image-container-inner');

			// grab the CSS duration from the computed class
			var cssDuration = curImg.css('transition-duration');
			if(cssDuration) {
				// if we support durations, delay rendering until we are done
				var delay = (cssDuration.toLowerCase().indexOf("ms")>-1) ? parseFloat(cssDuration) : parseFloat(cssDuration)*1000;
				if(!isNaN(delay)) {
					_.delay(_.bind(this._swapImages, this), Math.ceil(delay)+1);
				}
				else {
					// if we can't parse the delay, swap images immediately
					this._swapImages();
				}
			} else {
				// remove the image directly if we don't support animations
				this._swapImages();
			}
		},

		close: function(evt) {
			evt.preventDefault();
			this.model = null;
			$(document).off('keydown.deckinspectah');
			this.remove();
		},

		previousImage: function(evt) {
			evt.preventDefault();
			this._setSelectedImage(this.selectedImage - 1);
		},

		nextImage: function(evt) {
			evt.preventDefault();
			this._setSelectedImage(this.selectedImage + 1);
		},

		keydownHandler: function(evt) {
			switch(evt.which) {

				case 37:
					this.previousImage(evt);
					break;

				case 39:
					this.nextImage(evt);
					break;

				case 27:
					this.close(evt);
					break;

			}
		},

		// private
		_getSelectedModel : function() {
			return this.model.at(this.selectedImage);
		},

		_setSelectedImage: function(imgNum) {
			var imgLength = this.model.length;
			var newSelectedImage = imgNum > 0 ? (imgNum > imgLength-1 ?  imgLength-1 : imgNum) : 0;

			// ignore a click on the "before" or "after" buttons if they are at the end or beginning
			if(this.selectedImage !== newSelectedImage) {
				this.selectedImage = newSelectedImage;
				this.updateImage();
			}
		},

		_swapImages : function() {
			// show the loading spinner
			this.$('.w-deck-inspectah-overlay').addClass('w-deck-inspectah-loading');

			// generate the new model and render it
			var newModel = this.model.at(this.selectedImage);
			this.imageView.changeModel(newModel);
			if(this.sidebarView.changeModel) this.sidebarView.changeModel(newModel);

			// attach the newly generated items
			this.$('.w-deck-inspectah-image-container-inner').html(this.imageView.el);
			this.$('.w-deck-inspectah-sidebar-inner').html(this.sidebarView.el);

			var img = this.$('img#deckInspectahImage');
			var imgObj = new Image();
			var callback = _.bind(this._showImage, this);

			// bind the callback to the succes AND error handler
			imgObj.onload = callback;
			imgObj.onerror = callback;
			// load the image
			imgObj.src = img.attr('src');
		},

		_showImage : function() {
			// remove loading animations / images
			this.$('.w-deck-inspectah-loading').removeClass('w-deck-inspectah-loading');
			this.$('.w-deck-inspectah-image-loading').removeClass('w-deck-inspectah-image-loading');

			// re-enable the controls
			this.$('.w-deck-inspectah-control').removeClass('disabled');

			// disable first / last control depending on which image we're on
			if(this.selectedImage === 0) {
				this.$('.w-deck-inspectah-control.previous').addClass('disabled');
			}
			if(this.selectedImage === this.model.length - 1) {
				this.$('.w-deck-inspectah-control.next').addClass('disabled');
			}
		}
	});

	return DeckInspectahView;
});