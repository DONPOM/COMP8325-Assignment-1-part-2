var fitCoveVideo = function(elem){
	var iFrame = elem;
	var aspectRatio = null;
	var actualPlayerWidth = iFrame.width();
	var topBarHeight = 0;


	if(iFrame.attr("width") != '100%'){
		//for backwards compatibility use hardcoded old values for player dimensions
		try{
			var styleWidth = 512;
			var styleHeight = 288;
			aspectRatio = styleWidth / styleHeight;
 
			iFrame.css({
				"max-width" : styleWidth,
				"max-height" : actualPlayerWidth * styleHeight / styleWidth > styleHeight ? styleHeight : actualPlayerWidth * styleHeight / styleWidth,
				"min-height" : actualPlayerWidth * styleHeight / styleWidth > styleHeight ? styleHeight : actualPlayerWidth * styleHeight / styleWidth
			});
		}catch(e){}
	}else{
		// the new player with width=100% and height=100%
		try{
			aspectRatio = 16/9;

			if(iFrame.attr('src').indexOf('topbar=true') != -1){
				topBarHeight = 40;
			}
			
			iFrame.css({
				"height" : actualPlayerWidth / aspectRatio + topBarHeight,
			});
			iFrame.parent().css({
				"height" : actualPlayerWidth / aspectRatio + topBarHeight,
			});
		}catch(e){}
	}
};

jQuery(document).ready(function(){

	jQuery(".partner-player-wrapper iframe").each(function(){
						
		fitCoveVideo(jQuery(this));
	});
	
});

jQuery(window).resize(function(){
	jQuery("iframe").each(function(){
		if(jQuery(this).attr("id") == "partnerPlayer"){
			fitCoveVideo(jQuery(this));
		}
	});
	
});

