
            var fixCarousel = function () {

                var carousels = document.querySelectorAll('.carousel');

                for (var i = 0; i < carousels.length; i++) {
                    var items = jQuery('.carousel .item')
                    var sizes = [];
                    var maxHeight = 0;
                    var maxWidth = 0;
                    for (var j = 0; j < items.length; j++) {
                        var classes = items[j].getAttribute('class');
                        classes = classes.replace('item', '').replace(/^\s+|\s+$/, '');
                        items[j].setAttribute('class', classes);
                        if (items[j].offsetHeight > maxHeight) maxHeight = items[j].offsetHeight;
                        if (items[j].offsetWidth > maxWidth) maxWidth = items[j].offsetWidth;

                        sizes.push({
                            item:items[j],
                            height:items[j].offsetHeight,
                            width:items[j].offsetWidth
                        });

                        classes = 'item ' + classes;
                        classes = classes.replace(/^\s+|\s+$/, '');
                        items[j].setAttribute('class', classes);
                    }

                    for (var k = 0; k < sizes.length; k++) {
                        var item = sizes[k].item;
                        var aElement = item.getElementsByTagName('a')[0];
                        var imgElement = jQuery(aElement).children('img')//aElement.getElementsByTagName('img')[0];

                        aElement.setAttribute('style', 'display: block; margin:auto auto;');
                        items[k].setAttribute('style', 'background-color: #272727; width: ' + maxWidth + 'px; height: ' + maxHeight + 'px');

                        var topBottomPadding = 0;
                        if (sizes[k].height < maxHeight) {
                            var topBottomPadding = (maxHeight - sizes[k].height) / 2;
                        }
                        imgElement.attr('style', 'display:block; margin:auto auto; padding: ' + topBottomPadding + 'px 0px');
                    }

                }
            };

            var clearStyles = function () {
                var carousels = document.querySelectorAll('.carousel');

                for (var i = 0; i < carousels.length; i++) {
                    var items = carousels[i].children[0].children;
                    for (var j = 0; j < items.length; j++) {
                        var aElement = items[j].getElementsByTagName('a')[0];
                        aElement.setAttribute('style', '');
                        items[j].setAttribute('style', '');
                        aElement.getElementsByTagName('img')[0].setAttribute('style', '');
                    }
                }
            };

            (function ($, undefined) {

                var myCarousel_K_imgs = []
                var myCarousel_K_loaded = []

                
                    myCarousel_K_imgs.push(null)
                    myCarousel_K_loaded.push(true)
                
                
                    myCarousel_K_imgs.push("http://pbs.bento.storage.s3.amazonaws.com/hostedbento-prod/filer_public/thinktv/images/pledge/TTV%20AUG%20Pledge%20Slides.jpg")
                    myCarousel_K_loaded.push(false)
                
                
                    myCarousel_K_imgs.push("http://pbs.bento.storage.s3.amazonaws.com/hostedbento-prod/filer_public/thinktv/images/pledge/TTV%20AUG%20Pledge%20Slides2.jpg")
                    myCarousel_K_loaded.push(false)
                
                
                    myCarousel_K_imgs.push("http://bento.cdn.pbs.org/hostedbento-prod/filer_public/thinktv/images/programs/featured/50Years.jpg")
                    myCarousel_K_loaded.push(false)
                
                
                    myCarousel_K_imgs.push("http://pbs.bento.storage.s3.amazonaws.com/hostedbento-prod/filer_public/thinktv/images/pledge/TTV%20AUG%20Pledge%20Slides3.jpg")
                    myCarousel_K_loaded.push(false)
                
                
                    myCarousel_K_imgs.push("http://pbs.bento.storage.s3.amazonaws.com/hostedbento-prod/filer_public/thinktv/images/programs/holiday/HolidaySlide.jpg")
                    myCarousel_K_loaded.push(false)
                
                
                    myCarousel_K_imgs.push("http://pbs.bento.storage.s3.amazonaws.com/hostedbento-prod/filer_public/thinktv/images/jpg/KidsFam/OliveandOtto.JPG")
                    myCarousel_K_loaded.push(false)
                
				 
                
                
                
                

                var showNoImage = function () {
                    var items = $('#myCarousel-K div.item');
                    $.each(items, function (i, item) {

                        var img = $(item).find('img');
                        if ($(img).attr('src').replace(/\s+/g, '').length == 0) {
                            $(img).attr('src', "http://bento.cdn.pbs.org/hostedbento-prod/filer_public/_bento_media/img/no-image-available.jpg");
                        }
                    });
                };

                $(document).ready(function () {
                    try{
                        var items = $('#myCarousel-K div.item').length;
                        if (items > 1) {
                            var left = $('<a />').addClass('carousel-control left').attr('href', '#myCarousel-K').attr('data-slide', 'prev');
                            var right = $('<a />').addClass('carousel-control right').attr('href', '#myCarousel-K').attr('data-slide', 'next');
                            $('#myCarousel-K').append(left);
                            $('#myCarousel-K').append(right);
                        }

                        showNoImage();
                    }catch(e){}


                });

                $(window).resize(function(){
                    try{
                        window.clearStyles();
                        window.fixCarousel();
                    }catch(e){}
                })

                $(window).load(function () {
                    try{

                        $('#myCarousel-K.carousel.slide .item a img').each(function(iter){
                            if(myCarousel_K_imgs[iter]){
                                var callback = function(elem){
                                    myCarousel_K_loaded[iter] = true;

                                    var allLoaded = true;
                                    for(var i=0; i<myCarousel_K_loaded.length; i++){
                                        if(!myCarousel_K_loaded[i]){
                                            allLoaded = false;
                                            break;
                                        }
                                    }

                                    if(allLoaded){
                                        window.clearStyles();
                                        window.fixCarousel();
                                    }
                                }
                                $(this).error(function(){
                                    callback($(this));
                                });

                                $(this).one("load",function(){
                                    //for IE
                                    callback(this)
                                })
                                .attr("src", myCarousel_K_imgs[iter])
                                .each(function() {
                                    //Cache fix for browsers that don't trigger .load()
                                    if(this.complete){
                                        callback(this);
                                    }
                                })

                            }
                        })


                        //start carousell
                        $('#myCarousel-K.carousel.slide').carousel()
                    }catch(e){}
                })
            })(jQuery);
        