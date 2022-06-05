
$j(document).ready(function() {
	$j('img[original]').lazyload({
		effect: 'fadeIn',
		placeholder: "http://www.petmountain.com/Public/Images/white.gif"
	});
});
