jQuery(function($) {
	jQuery.fn.wCarousel = function(options) {

		var defaults = {
			carouselHolder: this.attr('id'),
			trackClicks: true
		}

		options = $.extend({}, defaults, options);
		var carousel = $('#' + options.carouselHolder);

		// carousel for all promo slides
		$('.slides', carousel).slides({
			play: 5000,
			hoverPause: true,
			animationStart: function() {
				if (options.carouselHolder == 'promo-carousel' || options.carouselHolder == 'logout-carousel') {
					// The slide does not get 'display:block' at the start of animation,
					// So, loop until it does.
					// If it never does, then it's the wrong slide, and animationCompelete()
					// will clear the interval
					social_interval = setInterval(social_render, 50);
				}
			},
			animationComplete: function() {
				if (options.carouselHolder == 'promo-carousel' || options.carouselHolder == 'logout-carousel') {
					clearInterval(social_interval);
				}
			}
		});

		if (options.carouselHolder == 'promo-carousel' || options.carouselHolder == 'logout-carousel') {
			// Some social buttons won't render when they are in in a hidden element,
			// So, we need to wait until the slide appears, and then render them.
			var social_interval, social_rendered = false;
			var social_slide = $('.social', carousel);
			var social_render = function() {
				// ditch if we've already done this
				if (social_rendered) {
					clearInterval(social_interval);
					return;
				}

				// wait until the slide is displayed
				if (social_slide.css('display') != 'block') return;

				social_rendered = true;
				clearInterval(social_interval);
				return;
				//$('.more-buttons', social_slide).append(promo_carousel_content); //Don't know what this was for, not defined in page.

				// re-run social button generator scripts
				if (typeof FB != 'undefined') FB.XFBML.parse();
				$.getScript('//platform.twitter.com/widgets.js');
			};

			social_render(); // in case the social slide is first
		}

		if (options.carouselHolder == 'promo-carousel') {
			// handle click analytics
			if (options.trackClicks) {
				$('.slide a', carousel).click(function() {
					var link = this;
					var slide = $(this).closest('.slide').attr('rel');

					if(jQuery(this).attr('target')) return true;

					$.get('/redirect.php', {
						'method': '\\Widget\\PromoCarousel::track_click',
						'params': Object.toJSON({
							'slide': slide
						})
					}, function() {
						if (this.href != '#') location.href = link.href;
					});

					return false;
				});
			}

			// countdown timer for coupon expiration
			$('.countdown', carousel).each(function() {
				var $this = $(this);
				var startTime = $this.text();
				$this.text('');
				$this.countdown({
					format: 'hh:mm:ss',
					startTime: startTime,
					digitWidth: 25,
					digitHeight: 35,
					image: "/images/digits.png"
				});
			});

			// submit buttons
			$('form a.submit', carousel).click(function() {
				$(this).closest('form').submit();
				return false;
			});

			// handle survey submissions
			$('form', carousel).submit(function() {
				// TODO: loading message

				var form = $(this);
				var slide = $(this).closest('.slide');

				$.post('/survey.php?s=2', form.serialize() + '&ajax=1', function() {
					if (!$(':input', form).val()) return false;

					var thanks = '.thanks';
					if ($('[name=how_likely]', form).val() < 4) thanks = '.sorry';
					thanks = $(thanks, slide);
					if (thanks.length) {
						thanks.show();
						form.remove();
					}
				});

				return false;
			});
		}

		return this;
	}
});