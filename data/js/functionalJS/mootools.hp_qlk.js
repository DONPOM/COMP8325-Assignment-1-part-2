var cutCoef = 1.5;
var twolineHeightOld = 0;
  
function swapQL(arg){
	//if (window.console) console.log(arg);
		ql_id=arg.replace('ql_','');
		$$('.quicklinksSlideRight').removeClass('ql_active');
		if ($('ql_div_'+ql_id)){
		
		$('ql_div_'+ql_id).addClass('ql_active');		
		//$('ql_img_'+ql_id).addClass('ql_active');
		}
}

function create_quick_slider(){
	sliderContainer=$('quicklinks-bar-slider')
	sliderObj=sliderContainer.getElement('ul');			
	default_position=parseInt(sliderObj.getStyle('marginLeft'))
		
	var myLeft = new Element('a', {
	    id: 'slider_left_cmd',
		'class': 'slider_nav',
		events: {
					click: function(){
					move_quick_slider('left');
					}
				}	
	});
	myLeft.inject($('quicklinks-bar-slider'),'top');
	
	var myRight = new Element('a', {
	    id: 'slider_right_cmd',
		'class': 'slider_nav',
		events: {
					click: function(){
					move_quick_slider('right');
					}
				}	
	});
  

  
	myRight.inject($('quicklinks-bar-slider'),'bottom');
	
	
	
	quick_slider_calculateIsReady();
	
	window.addEvent('changeSize',function(){
		quick_slider_calculateIsReady();		
    });	
		
	
}	

function quick_slider_calculateIsReady(){
	twolineHeight=parseInt($('quicklinks-bar-slider').offsetHeight);
	//if (window.console) console.log(twolineHeight) ;
	if (twolineHeightOld != twolineHeight){
	
		twolineHeightOld=twolineHeight;
		quick_slider_calculate()
	}else{
		
		setTimeout('quick_slider_calculateIsReady()',200)
	}
}

function quick_slider_calculate(){

	


	$$('#quicklinks-bar-slider li').each(function(el,index){
				el.setStyle('width',null);
				
				el.setStyle('width',Math.ceil(parseInt((el.getStyle('width')))/cutCoef));  
				el.setStyle('overflow','scroll');				
						
				if (parseInt(el.getElement('a').offsetHeight) > twolineHeight){
				
					while (parseInt(el.getElement('a').offsetHeight) > twolineHeight && parseInt(el.getStyle('width')) < $('quicklinks-bar-slider').offsetWidth){	
					
						el.setStyle('width',parseInt(el.getStyle('width'))+1); 
					
					}
					el.setStyle('width',parseInt(el.getStyle('width'))-12);// scrollbar width fix
				}else{
					el.setStyle('width',el.scrollWidth-parseInt(el.getStyle('paddingRight')));		
					
				}
				el.setStyle('overflow',null);				
	});
	quick_slider_init()
}

function quick_slider_init(){
		
		sliderObj.setStyle('marginLeft',default_position)
		nb_px=0;	
		li_list=sliderObj.getElements('li');
				
		li_pad_mar_bor=parseInt(li_list[0].getStyle('paddingLeft'))+parseInt(li_list[0].getStyle('paddingRight'))+parseInt(li_list[0].getStyle('marginLeft'))+parseInt(li_list[0].getStyle('marginRight'))+parseInt(li_list[0].getStyle('borderLeftWidth'))+parseInt(li_list[0].getStyle('borderRightWidth'));
		
		li_list.each(function(el,index){
			nb_px+=parseInt(el.getStyle('width'))+li_pad_mar_bor;  
			//if (window.console) console.log(nb_px)
		}); 
		
		slider_left_cmd_size=parseInt($('slider_left_cmd').getStyle('width'))+parseInt($('slider_left_cmd').getStyle('paddingLeft'))+parseInt($('slider_left_cmd').getStyle('paddingRight'))+parseInt($('slider_left_cmd').getStyle('marginLeft'))+parseInt($('slider_left_cmd').getStyle('marginRight'))+parseInt($('slider_left_cmd').getStyle('borderLeftWidth'))+parseInt($('slider_left_cmd').getStyle('borderRightWidth'));
		//if (window.console) console.log('slider_left_cmd_size='+slider_left_cmd_size)
		
		slider_right_cmd_size=parseInt($('slider_right_cmd').getStyle('width'))+parseInt($('slider_right_cmd').getStyle('paddingLeft'))+parseInt($('slider_right_cmd').getStyle('paddingRight'))+parseInt($('slider_right_cmd').getStyle('marginLeft'))+parseInt($('slider_right_cmd').getStyle('marginRight'))+parseInt($('slider_right_cmd').getStyle('borderLeftWidth'))+parseInt($('slider_right_cmd').getStyle('borderRightWidth'));
		//if (window.console) console.log('slider_right_cmd_size='+slider_right_cmd_size)

		
		nb_item_swap='max';// number or 'max'
		//if (window.console) console.log('width='+parseInt(sliderContainer.getStyle('width')))
		nb_px_visible = parseInt(sliderContainer.getStyle('width'))-(/*slider_left_cmd_size+*/slider_right_cmd_size);
		//if (window.console) console.log('nb_px_visible='+nb_px_visible)
		duration = 200;
		moving = 0;
		
		
		$('slider_left_cmd').addClass('hidden');
		if(nb_px <= nb_px_visible)
			{
			$('slider_right_cmd').addClass('hidden');	
			}
			
}	

function move_quick_slider(orientation){
		
			
		//if (window.console) console.log('nb_px_visible='+nb_px_visible)
		startPos=parseInt(sliderObj.getStyle('marginLeft'))-default_position;
		
		if (orientation == 'left'){
			limit= Math.abs(startPos)-nb_px_visible
			
		}else{
		   limit= Math.abs(startPos)+nb_px_visible
		}
		//if (window.console) console.log('limit='+limit)
	
		nextDelta=0;
		newtTarget=nextDelta;
		li_list.each(function(el,index){
		
			nextDelta+=parseInt(el.getStyle('width'))+li_pad_mar_bor;
			if (nextDelta<limit){
				newtTarget=nextDelta;
				
			}else{
			
				return false;
			}
			//if (window.console) console.log(newtTarget)
		});
		
		$('slider_left_cmd').removeClass('hidden');
		$('slider_right_cmd').removeClass('hidden');
		
		if (newtTarget <= default_position){
			newtTarget=0;
			$('slider_left_cmd').addClass('hidden');
			
		}else if (newtTarget > nb_px-nb_px_visible){
			newtTarget=nb_px-nb_px_visible;
			$('slider_right_cmd').addClass('hidden');
		}
		
		newtTarget=default_position-newtTarget;
	
				
		var sliding_fx = new Fx.Tween(sliderObj, {
			duration: duration,
			transition: Fx.Transitions.Sine.easeOut,
			onStart: function(){moving = 1;},
			onComplete: function(){moving = 0;}
			});
		
		sliding_fx.start('marginLeft', parseInt(sliderObj.getStyle('marginLeft')),newtTarget );
		
		
			
}		

window.addEvent('domready', function(){

	if ($('quicklinks-bar')){
		$('quicklinks-bar').addClass('js_on')
		create_quick_slider()
	}
	
});


