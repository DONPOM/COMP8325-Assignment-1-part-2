
            jQuery('.sale-arrow.prev').addClass('disabled');
            jQuery('.carousel').iosSlider({
                desktopClickDrag: true,
                snapToChildren: true,
                infiniteSlider: false,
                navNextSelector: '.sale-arrow.next',
                navPrevSelector: '.sale-arrow.prev',
                lastSlideOffset: 3,
                onFirstSlideComplete: function(){
                    jQuery('.sale-arrow.prev').addClass('disabled');
                },
                onLastSlideComplete: function(){
                    jQuery('.sale-arrow.next').addClass('disabled');
                },
                onSlideChange: function(){
                    jQuery('.sale-arrow.prev').removeClass('disabled');
                    jQuery('.sale-arrow.next').removeClass('disabled');
                }
            });
        