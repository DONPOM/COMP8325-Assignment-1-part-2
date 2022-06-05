/* Turn a form textbox into a "5 star" rating input. */

(function($) {
	$.fn.rating = function(options) {
		var options = $.extend({
			count: 5,
			submitOnClick: true,
			click: false,
			submit: false,
			low: "not likely",
			high: "very likely"
		}, options);

		return this.each(function() {
			var input = $(this);

			input.css({
				'visibility': 'hidden',
				'position': 'absolute'
			});

			var rating = $('<div class="rating"></div>').insertAfter(input);

			for (var ndx = 1; ndx <= options.count; ndx++) {
				$('<a href="#">' + ndx + '</a>').appendTo(rating);
			}

			$(
				'<p>' +
				'<span class="low">' + options.low + '</span>' +
				'<span class="high">' + options.high + '</span>' +
				'</p>'
			).appendTo(rating);

			$('a', rating)
				.hover(
					function(e) {
						var test_index = e.type == 'mouseenter'
							? $(this).index()
							: (parseInt(input.val(), 10) || 0) - 1;

						$('a', rating)
							.removeClass('on')
							.addClass(function(index) {
								return index <= test_index ? 'on' : '';
							});
					}
				)
				.click(function() {
					input.val($(this).text());
					if (options.submitOnClick) input.closest('form').submit();
					return false;
				});

			if (options.click) $('a', rating).click(options.click);
			if (options.submit) input.closest('form').submit(options.submit);
		});
	};
})(jQuery);