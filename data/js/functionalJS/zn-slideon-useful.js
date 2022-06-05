
    if(slide_on){
        jQuery(document).ready(function($) {
            $('.iosSlider').iosSlider({
                desktopClickDrag: true,
                touchMoveThreshold:4,
                snapToChildren: true,
                infiniteSlider: true,
                autoSlide:true,
                autoSlideTimer:8000,
        		navSlideSelector: '.sliderNavi .naviItem',                
                navNextSelector: '.iosSlider .next',
                navPrevSelector: '.iosSlider .prev',
        		onSlideChange: slideContentChange,
        		onSlideComplete: slideContentComplete,
        		onSliderLoaded: slideContentLoaded
            }); 
                       	
        }); 
    }
