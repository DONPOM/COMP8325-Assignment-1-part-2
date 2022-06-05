
IncludeJavaScript("/Public/Scripts/fancybox/fancybox.2.1.5/jquery.fancybox.pack.js",function() {
	jQuery(document).ready(function() {
		jQuery(".reviewM").fancybox({
			'padding'	: '0',
			'centerOnScroll'	:'true',
			'height'			:'600',
			'overlayOpacity'	:0.5, 
			'overlayColor'		:'#000000',
			'showCloseButton'	:true
		});
		jQuery("#welcometrigPM2").fancybox({
	'padding'		:	0,
	'centerOnScroll'	:	'true',
	'overlayOpacity':	0.5,
	'overlayColor'	:	'#000000'
});


if (jQuery("#welcomeMPM2").attr('id')) {
	if(jQuery.cookie('p_first') != "viewed") {
		jQuery.cookie('p_first',"viewed", { path: '/', expires:3650 });
		if (jQuery.cookie('p_first') == "viewed") {
			jQuery("#welcometrigPM2").trigger('click');
		}
	}
}

$j('.product_video_mod a').fancybox({
	'padding'			:0,
	'centerOnScroll'	:'true',
	'overlayColor'		:'#000000',
	'showCloseButton'	:true
});
$j('.reviewM').fancybox({
	'padding'	: '0',
	'centerOnScroll'	:'true',
	'width'				:'480',
	'overlayOpacity'	:0.5, 
	'overlayColor'		:'#000000',
	'showCloseButton'	:true
});	});
});
