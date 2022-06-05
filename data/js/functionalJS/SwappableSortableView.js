define(['jquery',
	'backbone'
], function ($, Backbone) {

	var SwappableSortableView = Backbone.View.extend({

		initialize: function () {
			this.$placeholder = this.options.$placeholder;
			this.$placeholder.addClass('sortablePlaceHolder');

			this.$moveableContainer = this.options.$moveableContainer;
			this.elementSelector = this.options.elementSelector;
			this.nonDragSelector = this.options.nonDragSelector;
			this.$scrollDiv = this.options.$scrollDiv;
			this.container = this.options.container;
			this.scrollTarget = this.options.scrollTarget;
			this.scrollSpacingTop = this.options.scrollSpacingTop;
			this.scrollSpacingBottom = this.options.scrollSpacingBottom;
		},

		start: function () {
			this.detach();
			this.$(this.elementSelector).on('mousedown', $.proxy(this.startDragging, this));
		},

		detach: function () {
			this.$(this.elementSelector).off('mousedown');
			this.$(this.nonDragSelector).off('mousedown');
		},

		startDragging: function (e) {
			if (this.$(e.target).is(this.nonDragSelector)) {
				return;
			}

			e.preventDefault();

			if (this.getTargetInfo(0, 0).placeHolderIndex === null) {
				this.$el.on('mousemove.swappableSortableView', e.currentTarget, $.proxy(this.drag, this));
				this.$el.on('mouseup.swappableSortableView', $.proxy(this.stopDragging, this));
			}
		},

		drag: function (e) {

			if (!this.$dragTarget) {
				this.trigger('dragStarted');
				this.$dragTarget = $(e.data);

				this.offset = {top: e.clientY - this.$dragTarget.offset().top + this.getScrollTop(),
					left: e.clientX - this.$dragTarget.offset().left};

				this.$placeholder.insertBefore(this.$dragTarget);
				this.$moveableContainer.empty().append(this.$dragTarget);
				this.$moveableContainer.show();
			}

			this.$moveableContainer.css('top', e.clientY - this.offset.top);
			this.$moveableContainer.css('left', e.clientX - this.offset.left);

			this.movePlaceHolder(e.clientY, e.clientX);
			this.dragEvent = e;

			if (!this.scrolling) {
				this.scroll();
			}
		},

		stopDragging: function (e) {
			this.$el.off('mousemove.swappableSortableView');
			this.$el.off('mouseup.swappableSortableView');

			var placeHolderIndex = this.getTargetInfo(0, 0).placeHolderIndex;
			this.$(this.getNthElement(placeHolderIndex))
				.before(this.$moveableContainer.children().first()).remove();

			this.$moveableContainer.html("");
			this.$moveableContainer.hide();

			this.$dragTarget = null;

			this.trigger("elementReordered");
		},

		movePlaceHolder: function (mouseTop, mouseLeft) {
			var targetInfo = this.getTargetInfo(mouseTop, mouseLeft);

			if (targetInfo.targetIndex === null) {
				return;
			}

			if (targetInfo.targetIndex != targetInfo.placeHolderIndex) {
				var dummyVar = null;
				var i = targetInfo.targetIndex;

				if (targetInfo.placeHolderIndex > targetInfo.targetIndex) { // Up
					for (; i < targetInfo.placeHolderIndex; i++) {
						if (targetInfo.targetIndex == i) {
							dummyVar = $('<div></div>').insertBefore(this.getNthElement(i));
						}
						this.getNthElement(i).insertBefore(this.getNthElement(i + 1));
					}

					this.$placeholder.insertBefore(dummyVar);
					dummyVar.remove();
				} else if (targetInfo.placeHolderIndex < targetInfo.targetIndex) {  // down
					for (; i > targetInfo.placeHolderIndex; i--) {
						if (targetInfo.targetIndex == i) {
							dummyVar = $('<div></div>').insertAfter(this.getNthElement(i));
						}
						this.getNthElement(i).insertAfter(this.getNthElement(i - 1));
					}

					this.$placeholder.insertAfter(dummyVar);
					dummyVar.remove();
				}
			}
		},

		getNthElement: function (index) {
			return this.$(this.elementSelector + ":eq(" + index + ")");
		},


		getTargetInfo: function (mouseTop, mouseLeft) {
			var targetIndex = null;
			var placeHolderIndex = null;

			var self = this;
			this.$(this.elementSelector).each(function (index) {
				var $element = $(this);
				var offset = $element.offset();

				if (offset.top <= mouseTop + self.getScrollTop() &&
					(offset.top + $element.outerHeight()) >= mouseTop + self.getScrollTop() &&
					offset.left <= mouseLeft && (offset.left + $element.outerWidth()) >= mouseLeft) {
					targetIndex = index;
				}

				if ($element.hasClass('sortablePlaceHolder')) {
					placeHolderIndex = index;
				}
			});

			return {
				targetIndex: targetIndex,
				placeHolderIndex: placeHolderIndex
			};
		},

		getScrollTop: function () {
			return  this.$scrollDiv.scrollTop();
		},

		scroll: function () {
			var self = this;
			this.scrolling = true;

			var containerTop = this.container.offset().top - this.getScrollTop();
			var containerBottomNotShown = (this.getScrollTop() + $(window).height()) - (this.container.offset().top + this.container.height());

			if (this.dragEvent &&
				this.dragEvent.clientY > ($(window).height() - (this.scrollTarget + this.scrollSpacingBottom) ) &&
				containerBottomNotShown < this.scrollSpacingBottom) { // Down

				this.$scrollDiv.animate({
					scrollTop: self.getScrollTop() + 100
				}, 100, 'linear', function () {
					setTimeout($.proxy(self.scroll, self));
				});
			} else if (this.dragEvent &&
				this.dragEvent.clientY < this.scrollTarget &&
				containerTop < this.scrollSpacingTop) {   // Up

				this.$scrollDiv.animate({
					scrollTop: self.getScrollTop() - 100
				}, 100, 'linear', function () {
					setTimeout($.proxy(self.scroll, self));
				});
			} else {
				this.scrolling = false;
			}
		}
	});

	return SwappableSortableView;
});
