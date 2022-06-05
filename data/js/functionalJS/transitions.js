define([
	'jquery'
], function($){
	var transitions = {};

	transitions.none = function(oldEl, newEl, duration, invert, callback) {
		callback();
	};

	transitions.puff = function(oldEl, newEl, duration, invert, callback) {
		oldEl.parent().css('overflow', 'visible');

		var resetCss = { opacity: '', top: 0, left: 0, width: '', height: '', 'z-index': '' };
		var cb = function(){
			oldEl.parent().css('overflow', '');
			oldEl.css(resetCss);
			newEl.css(resetCss);
			callback();
		};

		if(invert) {
			oldEl.css('z-index', 0);
			newEl.css({
				'z-index': 1,
				opacity: 0,
				top: '-25%',
				left: '-25%',
				width: '150%',
				height: '150%'
			}).animate({
				opacity: 1,
				top: '0%',
				left: '0%',
				width: '100%',
				height: '100%'
			}, duration, 'swing', cb);
		} else {
			newEl.css('z-index', 0);
			oldEl.css({
				'z-index': 1,
				opacity: 1,
				width: '100%',
				height: '100%',
				top: '0%',
				left: '0%'
			}).animate({
				opacity: 0,
				top: '-25%',
				left: '-25%',
				width: '150%',
				height: '150%'
			}, duration, 'swing', cb);
		}
	};

	transitions.fade = function(oldEl, newEl, duration, invert, callback) {
		oldEl.css({ 'z-index': 2, opacity: 1 });
		newEl.css({ 'z-index': 1, opacity: 1 });
		oldEl.animate({opacity: 0}, duration, 'swing', function() {
			oldEl.css({'z-index': '', opacity: ''});
			newEl.css({'z-index': '', opacity: ''});
			if(typeof callback === 'function') callback();
		});
	};

	transitions.slideHorizontal = function(oldEl, newEl, duration, invert, callback) {
		var resetCallback = function() {
			oldEl.css({'z-index': '', opacity: '', 'left': 0});
			newEl.css({'z-index': '', opacity: '', 'left': 0});
			if(typeof callback === 'function') callback();
		};
		if(invert) {
			oldEl.css({'z-index': 2, 'left': '0%'}).animate({left: '100%'}, duration, 'swing', resetCallback);
			newEl.css('z-index', 1);
		} else {
			newEl.css({'z-index': 2, 'left': '100%'}).animate({left: '0%'}, duration, 'swing', resetCallback);
			oldEl.css('z-index', 1);
		}
	};

	transitions.slideVertical = function(oldEl, newEl, duration, invert, callback) {
		var resetCallback = function() {
			oldEl.css({'z-index': '', opacity: '', 'top': 0});
			newEl.css({'z-index': '', opacity: '', 'top': 0});
			if(typeof callback === 'function') callback();
		};
		if(invert) {
			newEl.css('z-index', 1);
			oldEl.css({'z-index':2}).animate({top: '-100%'}, duration, 'swing', resetCallback);
		} else {
			newEl.css({'z-index': 2, top: '-200%'}).animate({top: 0}, duration, 'swing', resetCallback);
			oldEl.css('z-index', 1);
		}
	};

	transitions.carouselHorizontal = function(oldEl, newEl, duration, invert, callback) {
		var resetCallback = function() {
			oldEl.css({'z-index': '', opacity: '', 'left': 0});
			newEl.css({'z-index': '', opacity: '', 'left': 0});
			if(typeof callback === 'function') callback();
		};
		var before = {left: '-100%'},
			current = {left: '0%'},
			after = {left: '100%'};
		if(invert) {
			oldEl.css(current).animate(after, duration, 'swing');
			newEl.css(before).animate(current, duration, 'swing', resetCallback);
		} else {
			oldEl.css(current).animate(before, duration, 'swing');
			newEl.css(after).animate(current, duration, 'swing', resetCallback);
		}
	};

	transitions.carouselVertical = function(oldEl, newEl, duration, invert, callback) {
		var resetCallback = function() {
			oldEl.css({'z-index': '', opacity: '', 'top': 0});
			newEl.css({'z-index': '', opacity: '', 'top': 0});
			if(typeof callback === 'function') callback();
		};
		var before = {top: '-100%'},
			current = {top: '0%'},
			after = {top: '100%'};
		if(invert) {
			oldEl.css(current).animate(after, duration, 'swing');
			newEl.css(before).animate(current, duration, 'swing', resetCallback);
		} else {
			oldEl.css(current).animate(before, duration, 'swing');
			newEl.css(after).animate(current, duration, 'swing', resetCallback);
		}
	};

	transitions.random = function(oldEl, newEl, duration, invert, callback) {
		var ts = $.map(transitions, function(transition, transitionName){
			if(transitionName !== 'random' && transitionName !== 'none') return transition;
		});
		ts[Math.floor(Math.random()*ts.length)](oldEl, newEl, duration, invert, callback);
	};

	if(!window.webs) window.webs = {};
	window.webs.transitions = transitions;

	return transitions;
});