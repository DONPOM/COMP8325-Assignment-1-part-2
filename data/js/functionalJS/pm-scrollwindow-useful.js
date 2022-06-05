
$j(window).scroll(function(){
	if( $j(window).scrollTop() > 300 ) {
			$j('.to-the-top.tc_ghost').removeClass('tc_ghost');
			$j('.to-the-top').fadeIn();
	} else {
			$j('.to-the-top').fadeOut();
	}
});
