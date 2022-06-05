$(document).ready(function() {

	$(".type_banner").click(function() {
		ga('send', 'event', { eventCategory: 'Popups', eventAction: 'Banners'});
	});

	$(".type_logo").click(function() {
		ga('send', 'event', { eventCategory: 'Popups', eventAction: 'Logo'});
	});

	$(".type_header_btn").click(function() {
		ga('send', 'event', { eventCategory: 'Popups', eventAction: 'Header-Btn-Join'});
	});

	$(".type_chat_btn").click(function() {
		ga('send', 'event', { eventCategory: 'Popups', eventAction: 'Video-Btn-Chat'});
	});

	$(".type_thumb").click(function() {
		ga('send', 'event', { eventCategory: 'Popups', eventAction: 'Thumbs'});
	});
	
	$(".type_category").click(function() {
		//ga('send', 'event', { eventCategory: 'Popups', eventAction: 'CategoryFilter'});
		var CategName = $(this).data('category');		
		ga('send', 'event', { eventCategory: 'Popups', eventAction: 'CategoryFilter', eventLabel: CategName });
	});	
	
	$(".type_next_model_btn").click(function() {
		ga('send', 'event', { eventCategory: 'Popups', eventAction: 'NextModel-Btn'});
	});		
	
	$("#livePreview").click(function() {
		ga('send', 'event', { eventCategory: 'Popups', eventAction: 'ClickVideoStream'});
	});		
	
	$("body").click(function (event){		
		var targets = ['IMG','A','OBJECT'];
		var targetName = event.target.nodeName;
		var hasLINK = (targets.indexOf(targetName) > -1);		
		if( !hasLINK ) {
			ga('send', 'event', { eventCategory: 'Popups', eventAction: 'BackgroundClick'});
			window.location.href=window.Plugin.Models[0].Url;			
		} 
	});

});	