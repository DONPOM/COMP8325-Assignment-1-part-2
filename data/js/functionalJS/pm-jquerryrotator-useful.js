
if ($j('.fcol1').length) {
	$j('.FP .name').html('<span>TOP PET SUPPLIES »</span>');
	$j('.CustomerReviews .name').html('<span>CUSTOMER REVIEWS »</span>');
	//
	IncludeJavaScript("/content/jquery_rotator.js",function(){
		$j('.rotes').jshowoff();
		$j('.jshowoff-prev').html('<img src="/content/promoArrow-l.png" border="0">');
		$j('.jshowoff-next').html('<img src="/content/promoArrow-r.png" border="0">');
	});
}
