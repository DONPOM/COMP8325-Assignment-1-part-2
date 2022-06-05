
$(window).load(function() {
	$('#scoreboard-ticker').find('.inProgress').css({'display': 'inline'});
		$('#scoreboard-ticker').easySlider({
		auto: true,
		continuous: true,
		controlsShow: true,
		vertical: false,
		nextText: '',
		prevText: '',
		prevId: 'scoreboard-ticker-prev',
		nextId: 'scoreboard-ticker-next',
		playPauseId: 'scoreboard-ticker-pp',
		playText: '',
		pauseText: '',
		speed: 1000,
		pause: 3000
	});
	});
