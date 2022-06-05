document.write('<style>#containerVideos .euBox{display:none;}.hiddenJSExec{display:none;}</style>')



window.addEvent('domready', function() {
		if(!isCssActivate()) {
		$$('.hiddenNoJS').each(function(el){
					el.dispose();
			});
	}else{
		
		$$('.dl_video').each(function(el){
				el.dispose();
			});
			
			
			$$('.SlideItMoo_element p strong a').each(function(el){
			//	alert(el.innerHTML)
				el.getParent('strong').innerHTML=el.innerHTML;
				
			});
					$$('.video_thumbnail').each(function(el,x){
			
										var mynewimg  = new Element('img', {'class': 'video_arrow','src': '/wel/images/eu_portal/video-arrow.gif','events': {
													'click': function(){
															displayvideo(x);															
															}
													 }

											});						
						
											mynewimg.inject(el,'after');
			
				
				el.addEvent('click',function(){
				displayvideo(x);
					
				});
				
	
			});
		}
	
	
	
			
			$$('.hiddenNoJS').each(function(el){
					el.removeClass('hiddenNoJS');
			});
			
	

			
			/*var gallery1 = new slideGallery($("SlideItMoo_outer"), {
				steps: 3,
				speed: 600,
				transition: Fx.Transitions.Sine.easeInOut,
				mode: "line"
			});*/
			
	
			
			});
			
			
			

function displayvideo(vid){
	$$('.mediaFrame').each(function(el){
		el.setStyle('display','none')
		}
	)	
	
	if (!$defined($('cl_'+vid))){
	
		var myFirstElement  = new Element('p', {
			'class': 'p_close',
			'id': 'cl_'+vid,
			'align': 'center',
			'html':'<a class="video_close" href="#containerVideos">'+closelabel+'</a>'
			,'events': {
	        'click': function(){
	           //alert(this);
	           $$('.mediaFrame')[vid].setStyle('display','none');
	        }
        }
			
			
			});
			myFirstElement.inject($$('.mediaFrame')[vid],'bottom');
		}
			
				
				$$('.mediaFrame')[vid].setStyle('display','block');			
}